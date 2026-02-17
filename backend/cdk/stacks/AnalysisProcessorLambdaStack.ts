import { Duration } from "aws-cdk-lib";
import { Construct } from "constructs";
import { Function } from "aws-cdk-lib/aws-lambda";
import {
  AWS_LAMBDA_LAYERS,
  BaseLambdaStack,
  BaseLambdaStackProps,
  INTEGRATION_LAMBDA_LAYERS,
  LambdaCodeRunTimeConfig,
} from "@wayweaver/ariadne";
import { Queue } from "aws-cdk-lib/aws-sqs";
import { SqsEventSource } from "aws-cdk-lib/aws-lambda-event-sources";
import { Table } from "aws-cdk-lib/aws-dynamodb";
import { Effect, PolicyStatement } from "aws-cdk-lib/aws-iam";
import { AwsConfig } from "../configs/AwsConfig";

export interface AnalysisProcessorLambdaStackProps extends BaseLambdaStackProps {
  analysisExecutionQueue: Queue;
  analysisExecutionsTable: Table;
}

export class AnalysisProcessorLambdaStack extends BaseLambdaStack {
  constructor(scope: Construct, id: string, props: AnalysisProcessorLambdaStackProps) {
    super(scope, id, props);
  }

  protected buildLambdaFunction(
    props: AnalysisProcessorLambdaStackProps,
    lambdaCodeRunTimeConfig: LambdaCodeRunTimeConfig
  ): Function {
    const secretAccessStatement = new PolicyStatement({
      effect: Effect.ALLOW,
      actions: ["secretsmanager:GetSecretValue"],
      resources: [AwsConfig.OPENAI_API_SECRET_ARN],
    });

    const analysisProcessorLambdaId = props.idBuilder.build("analysis-processor-lambda");
    const analysisProcessorLambda = new Function(this, analysisProcessorLambdaId, {
      ...lambdaCodeRunTimeConfig,
      ...this.buildLambdaLayersConfig([
        AWS_LAMBDA_LAYERS.AWS_QUEUE_LAMBDA_LAYER,
        AWS_LAMBDA_LAYERS.AWS_DYNAMODB_LAMBDA_LAYER,
        AWS_LAMBDA_LAYERS.AWS_OPENSEARCH_LAMBDA_LAYER,
        INTEGRATION_LAMBDA_LAYERS.LANGCHAIN_LANGGRAPH_LAMBDA_LAYER,
      ]),
      functionName: analysisProcessorLambdaId,
      description: "Lambda function to process analysis execution",
      handler: "AnalysisProcessorLambda.handler",
      environment: {
        ANALYSIS_EXECUTIONS_TABLE: props.analysisExecutionsTable.tableName,
        OPENAI_API_SECRET_ID: AwsConfig.OPENAI_API_SECRET_ID,
      },
      timeout: AwsConfig.QUEUE_TIMEOUT_MINS,
      memorySize: 1024,
    });

    analysisProcessorLambda.addToRolePolicy(secretAccessStatement);
    analysisProcessorLambda.addEventSource(
      new SqsEventSource(props.analysisExecutionQueue, {
        batchSize: 1,
        maxBatchingWindow: Duration.seconds(5),
      })
    );
    props.analysisExecutionsTable.grantReadWriteData(analysisProcessorLambda);

    return analysisProcessorLambda;
  }
}
