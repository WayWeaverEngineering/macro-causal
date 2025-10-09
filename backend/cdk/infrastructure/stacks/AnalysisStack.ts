import { Stack, Duration, RemovalPolicy } from "aws-cdk-lib";
import * as path from 'path';
import { Construct } from "constructs";
import { DefaultIdBuilder } from "../../utils/Naming";
import { Queue } from "aws-cdk-lib/aws-sqs";
import { Code as LambdaCode, Function as LambdaFunction } from "aws-cdk-lib/aws-lambda"
import { SqsEventSource } from "aws-cdk-lib/aws-lambda-event-sources";
import { Role, ServicePrincipal, ManagedPolicy, PolicyStatement, Effect } from "aws-cdk-lib/aws-iam";
import { Table, AttributeType, BillingMode, ProjectionType } from "aws-cdk-lib/aws-dynamodb";
import { AWS_QUEUE_LAMBDA_LAYER_NAME, COMMON_UTILS_LAMBDA_LAYER_NAME, DEFAULT_LAMBDA_NODEJS_RUNTIME, AWS_DYNAMODB_LAMBDA_LAYER_NAME, LANGCHAIN_LANGGRAPH_LAMBDA_LAYER_NAME, AWS_OPENSEARCH_LAMBDA_LAYER_NAME, PrebuiltLambdaLayersStack } from "@wayweaver/ariadne";
import { LambdaConfig } from "../configs/LambdaConfig";
import { AwsConfig } from "../configs/AwsConfig";

export interface AnalysisStackProps {
  lambdaLayersStack: PrebuiltLambdaLayersStack;
}

export class AnalysisStack extends Stack {
  public readonly analysisExecutionsTable: Table;
  public readonly analysisQueue: Queue;
  public readonly analysisSchedulingLambda: LambdaFunction;
  public readonly analysisProcessorLambda: LambdaFunction;
  public readonly analysisStatusLambda: LambdaFunction;

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
      visibilityTimeout: AwsConfig.QUEUE_TIMEOUT_MINS,
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
    const openSearchIntegrationLambdaLayer = props.lambdaLayersStack.getLayer(AWS_OPENSEARCH_LAMBDA_LAYER_NAME)
    const dynamodbIntegrationLambdaLayer = props.lambdaLayersStack.getLayer(AWS_DYNAMODB_LAMBDA_LAYER_NAME)
    const awsQueueLambdaLayer = props.lambdaLayersStack.getLayer(AWS_QUEUE_LAMBDA_LAYER_NAME)


    // IAM policy statement to allow pipeline stages to fetch secrets from Secret Manager
    const secretAccessStatement = new PolicyStatement({
      effect: Effect.ALLOW,
      actions: ['secretsmanager:GetSecretValue'],
      resources: [
        AwsConfig.OPENAI_API_SECRET_ARN
      ]
    })
    // Create IAM role for Lambda functions
    const lambdaRole = new Role(this, DefaultIdBuilder.build('analysis-lambda-role'), {
      assumedBy: new ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [
        ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
        ManagedPolicy.fromAwsManagedPolicyName('AmazonDynamoDBFullAccess'),
        ManagedPolicy.fromAwsManagedPolicyName('AmazonSQSFullAccess')
      ]
    });

    // Add secret access statement to scheduling lambda role
    lambdaRole.addToPrincipalPolicy(secretAccessStatement);

    // Create AnalysisSchedulingLambda
    const schedulingLambdaId = DefaultIdBuilder.build('analysis-scheduling-lambda');
    this.analysisSchedulingLambda = new LambdaFunction(this, schedulingLambdaId, {
      functionName: schedulingLambdaId,
      runtime: DEFAULT_LAMBDA_NODEJS_RUNTIME,
      code: LambdaCode.fromAsset(path.join(__dirname, LambdaConfig.LAMBDA_CODE_RELATIVE_PATH)),
      handler: 'AnalysisSchedulingLambda.handler',
      timeout: AwsConfig.QUEUE_TIMEOUT_MINS,
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
    this.analysisProcessorLambda = new LambdaFunction(this, processorLambdaId, {
      functionName: processorLambdaId,
      runtime: DEFAULT_LAMBDA_NODEJS_RUNTIME,
      code: LambdaCode.fromAsset(path.join(__dirname, LambdaConfig.LAMBDA_CODE_RELATIVE_PATH)),
      handler: 'AnalysisProcessorLambda.handler',
      timeout: AwsConfig.QUEUE_TIMEOUT_MINS,
      memorySize: 1024,
      role: lambdaRole,
      environment: {
        ANALYSIS_EXECUTIONS_TABLE: this.analysisExecutionsTable.tableName,
        OPENAI_API_SECRET_ID: AwsConfig.OPENAI_API_SECRET_ID
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
    this.analysisStatusLambda = new LambdaFunction(this, statusLambdaId, {
      functionName: statusLambdaId,
      runtime: DEFAULT_LAMBDA_NODEJS_RUNTIME,
      code: LambdaCode.fromAsset(path.join(__dirname, LambdaConfig.LAMBDA_CODE_RELATIVE_PATH)),
      handler: 'AnalysisStatusLambda.handler',
      timeout: AwsConfig.QUEUE_TIMEOUT_MINS,
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
  }
}
