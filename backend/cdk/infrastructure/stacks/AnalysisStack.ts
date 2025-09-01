import { Stack, Duration, RemovalPolicy } from "aws-cdk-lib";
import { Construct } from "constructs";
import { DefaultIdBuilder } from "../../utils/Naming";
import { Queue } from "aws-cdk-lib/aws-sqs";
import { Function, Runtime, Code, Architecture } from "aws-cdk-lib/aws-lambda";
import { SqsEventSource } from "aws-cdk-lib/aws-lambda-event-sources";
import { RestApi, LambdaIntegration, Cors } from "aws-cdk-lib/aws-apigateway";
import { Role, ServicePrincipal, ManagedPolicy } from "aws-cdk-lib/aws-iam";
import { Table, AttributeType, BillingMode, ProjectionType } from "aws-cdk-lib/aws-dynamodb";
import { AWS_QUEUE_LAMBDA_LAYER_NAME, COMMON_UTILS_LAMBDA_LAYER_NAME, DYNAMODB_INTEGRATION_LAMBDA_LAYER_NAME, LANGCHAIN_LANGGRAPH_LAMBDA_LAYER_NAME, OPEN_SEARCH_INTEGRATION_LAMBDA_LAYER_NAME, PrebuiltLambdaLayersStack } from "@wayweaver/ariadne";

export interface AnalysisStackProps {
  lambdaLayersStack: PrebuiltLambdaLayersStack;
}

export class AnalysisStack extends Stack {
  public readonly analysisExecutionsTable: Table;
  public readonly analysisQueue: Queue;
  public readonly analysisSchedulingLambda: Function;
  public readonly analysisProcessorLambda: Function;
  public readonly analysisStatusLambda: Function;
  public readonly analysisApi: RestApi;

  constructor(scope: Construct, id: string, props: AnalysisStackProps) {
    super(scope, id);

    // Create DynamoDB table for tracking analysis executions
    const tableId = DefaultIdBuilder.build('analysis-executions-table');
    this.analysisExecutionsTable = new Table(this, tableId, {
      tableName: tableId,
      partitionKey: { name: 'executionId', type: AttributeType.STRING },
      billingMode: BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.DESTROY, // For development/demo
      pointInTimeRecovery: false, // Disabled for demo to reduce costs,
    });

    // Add a GSI for querying by sessionId if needed in the future
    this.analysisExecutionsTable.addGlobalSecondaryIndex({
      indexName: 'sessionId-index',
      partitionKey: { name: 'sessionId', type: AttributeType.STRING },
      sortKey: { name: 'createdAt', type: AttributeType.STRING },
      projectionType: ProjectionType.ALL
    });

    // Add a GSI for querying by status if needed for monitoring
    this.analysisExecutionsTable.addGlobalSecondaryIndex({
      indexName: 'status-index',
      partitionKey: { name: 'status', type: AttributeType.STRING },
      sortKey: { name: 'createdAt', type: AttributeType.STRING },
      projectionType: ProjectionType.ALL
    });

    // Create SQS queue for analysis execution messages
    const queueId = DefaultIdBuilder.build('analysis-executions-queue');
    this.analysisQueue = new Queue(this, queueId, {
      queueName: queueId,
      visibilityTimeout: Duration.seconds(300), // 5 minutes
      retentionPeriod: Duration.days(14),
      deadLetterQueue: {
        queue: new Queue(this, DefaultIdBuilder.build('analysis-executions-dlq'), {
          queueName: DefaultIdBuilder.build('analysis-executions-dlq'),
          retentionPeriod: Duration.days(14)
        }),
        maxReceiveCount: 3
      }
    });

    const commonUtilsLambdaLayer = props.lambdaLayersStack.getLayer(COMMON_UTILS_LAMBDA_LAYER_NAME)
    const langChainLangGraphLambdaLayer = props.lambdaLayersStack.getLayer(LANGCHAIN_LANGGRAPH_LAMBDA_LAYER_NAME)
    const openSearchIntegrationLambdaLayer = props.lambdaLayersStack.getLayer(OPEN_SEARCH_INTEGRATION_LAMBDA_LAYER_NAME)
    const dynamodbIntegrationLambdaLayer = props.lambdaLayersStack.getLayer(DYNAMODB_INTEGRATION_LAMBDA_LAYER_NAME)
    const awsQueueLambdaLayer = props.lambdaLayersStack.getLayer(AWS_QUEUE_LAMBDA_LAYER_NAME)

    // Create IAM role for Lambda functions
    const lambdaRole = new Role(this, DefaultIdBuilder.build('analysis-lambda-role'), {
      assumedBy: new ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [
        ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
        ManagedPolicy.fromAwsManagedPolicyName('AmazonDynamoDBFullAccess'),
        ManagedPolicy.fromAwsManagedPolicyName('AmazonSQSFullAccess')
      ]
    });

    // Create AnalysisSchedulingLambda
    const schedulingLambdaId = DefaultIdBuilder.build('analysis-scheduling-lambda');
    this.analysisSchedulingLambda = new Function(this, schedulingLambdaId, {
      functionName: schedulingLambdaId,
      runtime: Runtime.NODEJS_18_X,
      architecture: Architecture.ARM_64,
      code: Code.fromAsset('../lambda/dist'),
      handler: 'AnalysisSchedulingLambda.handler',
      timeout: Duration.seconds(30),
      memorySize: 256,
      role: lambdaRole,
      environment: {
        ANALYSIS_EXECUTIONS_TABLE: this.analysisExecutionsTable.tableName,
        ANALYSIS_EXECUTIONS_QUEUE_URL: this.analysisQueue.queueUrl,
      },
      layers: [
        awsQueueLambdaLayer,
        commonUtilsLambdaLayer,
        dynamodbIntegrationLambdaLayer
      ]
    });

    // Grant DynamoDB write permissions
    this.analysisExecutionsTable.grantWriteData(this.analysisSchedulingLambda);
    
    // Grant SQS send permissions
    this.analysisQueue.grantSendMessages(this.analysisSchedulingLambda);

    // Create AnalysisProcessorLambda
    const processorLambdaId = DefaultIdBuilder.build('analysis-processor-lambda');
    this.analysisProcessorLambda = new Function(this, processorLambdaId, {
      functionName: processorLambdaId,
      runtime: Runtime.NODEJS_18_X,
      architecture: Architecture.ARM_64,
      code: Code.fromAsset('../lambda/dist'),
      handler: 'AnalysisProcessorLambda.handler',
      timeout: Duration.seconds(300), // 5 minutes
      memorySize: 1024,
      role: lambdaRole,
      environment: {
        ANALYSIS_EXECUTIONS_TABLE: this.analysisExecutionsTable.tableName,
      },
      layers: [
        awsQueueLambdaLayer,
        commonUtilsLambdaLayer,
        langChainLangGraphLambdaLayer,
        openSearchIntegrationLambdaLayer,
        dynamodbIntegrationLambdaLayer
      ]
    });

    // Grant DynamoDB read/write permissions
    this.analysisExecutionsTable.grantReadWriteData(this.analysisProcessorLambda);

    // Add SQS event source to processor lambda
    this.analysisProcessorLambda.addEventSource(
      new SqsEventSource(this.analysisQueue, {
        batchSize: 1, // Process one message at a time for better error handling
        maxBatchingWindow: Duration.seconds(5)
      })
    );

    // Create AnalysisStatusLambda
    const statusLambdaId = DefaultIdBuilder.build('analysis-status-lambda');
    this.analysisStatusLambda = new Function(this, statusLambdaId, {
      functionName: statusLambdaId,
      runtime: Runtime.NODEJS_18_X,
      architecture: Architecture.ARM_64,
      code: Code.fromAsset('../lambda/dist'),
      handler: 'AnalysisStatusLambda.handler',
      timeout: Duration.seconds(30),
      memorySize: 256,
      role: lambdaRole,
      environment: {
        ANALYSIS_EXECUTIONS_TABLE: this.analysisExecutionsTable.tableName,
      },
      layers: [
        commonUtilsLambdaLayer,
        dynamodbIntegrationLambdaLayer
      ]
    });

    // Grant DynamoDB read permissions
    this.analysisExecutionsTable.grantReadData(this.analysisStatusLambda);

    // Create API Gateway
    const apiId = DefaultIdBuilder.build('analysis-api');
    this.analysisApi = new RestApi(this, apiId, {
      restApiName: apiId,
      description: 'API for analysis operations',
      defaultCorsPreflightOptions: {
        allowOrigins: Cors.ALL_ORIGINS,
        allowMethods: Cors.ALL_METHODS,
        allowHeaders: ['Content-Type', 'X-Amz-Date', 'Authorization', 'X-Api-Key'],
        maxAge: Duration.days(1)
      }
    });

    // Create API resources and methods
    const analysisResource = this.analysisApi.root.addResource('analysis');
    
    // POST /analysis - Schedule new analysis
    analysisResource.addMethod('POST', new LambdaIntegration(this.analysisSchedulingLambda));
    
    // GET /analysis/{executionId} - Get analysis status
    const executionResource = analysisResource.addResource('{executionId}');
    executionResource.addMethod('GET', new LambdaIntegration(this.analysisStatusLambda));

    // Output important values
    this.exportValue(this.analysisQueue.queueUrl, {
      name: DefaultIdBuilder.build('analysis-queue-url')
    });

    this.exportValue(this.analysisApi.url, {
      name: DefaultIdBuilder.build('analysis-api-url')
    });
  }
}
