import { Stack, Duration, RemovalPolicy } from "aws-cdk-lib";
import * as path from "path";
import { Construct } from "constructs";
import { Queue } from "aws-cdk-lib/aws-sqs";
import { Function as LambdaFunction } from "aws-cdk-lib/aws-lambda";
import { Table, AttributeType, BillingMode, ProjectionType } from "aws-cdk-lib/aws-dynamodb";
import { PrebuiltLambdaLayers, ConstructIdBuilder } from "@wayweaver/ariadne";
import { LambdaConfig } from "../configs/LambdaConfig";
import { AwsConfig } from "../configs/AwsConfig";
import { AnalysisSchedulingLambdaStack } from "./AnalysisSchedulingLambdaStack";
import { AnalysisProcessorLambdaStack } from "./AnalysisProcessorLambdaStack";
import { AnalysisStatusLambdaStack } from "./AnalysisStatusLambdaStack";

export interface AnalysisStackProps {
  idBuilder: ConstructIdBuilder;
  prebuiltLambdaLayers: PrebuiltLambdaLayers;
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
    const tableId = props.idBuilder.build("analysis-executions-table");
    this.analysisExecutionsTable = new Table(this, tableId, {
      tableName: tableId,
      partitionKey: { name: "executionId", type: AttributeType.STRING },
      billingMode: BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.DESTROY, // For development/demo
      pointInTimeRecovery: false, // Disabled for demo to reduce costs,
    });

    // Add a GSI for querying by sessionId if needed in the future
    this.analysisExecutionsTable.addGlobalSecondaryIndex({
      indexName: "sessionId-index",
      partitionKey: { name: "sessionId", type: AttributeType.STRING },
      sortKey: { name: "createdAt", type: AttributeType.STRING },
      projectionType: ProjectionType.ALL,
    });

    // Add a GSI for querying by status if needed for monitoring
    this.analysisExecutionsTable.addGlobalSecondaryIndex({
      indexName: "status-index",
      partitionKey: { name: "status", type: AttributeType.STRING },
      sortKey: { name: "createdAt", type: AttributeType.STRING },
      projectionType: ProjectionType.ALL,
    });

    // Create SQS queue for analysis execution messages
    const queueId = props.idBuilder.build("analysis-executions-queue");
    this.analysisQueue = new Queue(this, queueId, {
      queueName: queueId,
      visibilityTimeout: AwsConfig.QUEUE_TIMEOUT_MINS,
      retentionPeriod: Duration.days(14),
      deadLetterQueue: {
        queue: new Queue(this, props.idBuilder.build("analysis-executions-dlq"), {
          queueName: props.idBuilder.build("analysis-executions-dlq"),
          retentionPeriod: Duration.days(14),
        }),
        maxReceiveCount: 3,
      },
    });

    const baseProps = {
      idBuilder: props.idBuilder,
      prebuiltLambdaLayers: props.prebuiltLambdaLayers,
      relativeLambdaCodePath: path.join(__dirname, LambdaConfig.LAMBDA_CODE_RELATIVE_PATH),
    };

    const schedulingStack = new AnalysisSchedulingLambdaStack(
      this,
      props.idBuilder.build("analysis-scheduling-lambda-stack"),
      {
        ...baseProps,
        analysisExecutionQueue: this.analysisQueue,
        analysisExecutionsTable: this.analysisExecutionsTable,
      }
    );

    const processorStack = new AnalysisProcessorLambdaStack(
      this,
      props.idBuilder.build("analysis-processor-lambda-stack"),
      {
        ...baseProps,
        analysisExecutionQueue: this.analysisQueue,
        analysisExecutionsTable: this.analysisExecutionsTable,
      }
    );

    const statusStack = new AnalysisStatusLambdaStack(
      this,
      props.idBuilder.build("analysis-status-lambda-stack"),
      {
        ...baseProps,
        analysisExecutionsTable: this.analysisExecutionsTable,
      }
    );

    this.analysisSchedulingLambda = schedulingStack.lambdaFunction;
    this.analysisProcessorLambda = processorStack.lambdaFunction;
    this.analysisStatusLambda = statusStack.lambdaFunction;
  }
}
