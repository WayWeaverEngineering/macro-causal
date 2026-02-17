import { Construct } from "constructs";
import { Function } from "aws-cdk-lib/aws-lambda";
import {
  AWS_LAMBDA_LAYERS,
  BaseLambdaStack,
  BaseLambdaStackProps,
  LambdaCodeRunTimeConfig,
} from "@wayweaver/ariadne";
import { Queue } from "aws-cdk-lib/aws-sqs";
import { Table } from "aws-cdk-lib/aws-dynamodb";
import { Effect, PolicyStatement } from "aws-cdk-lib/aws-iam";
import { AwsConfig } from "../configs/AwsConfig";

export interface AnalysisSchedulingLambdaStackProps extends BaseLambdaStackProps {
  analysisExecutionQueue: Queue;
  analysisExecutionsTable: Table;
}

export class AnalysisSchedulingLambdaStack extends BaseLambdaStack {
  constructor(scope: Construct, id: string, props: AnalysisSchedulingLambdaStackProps) {
    super(scope, id, props);
  }

  protected buildLambdaFunction(
    props: AnalysisSchedulingLambdaStackProps,
    lambdaCodeRunTimeConfig: LambdaCodeRunTimeConfig
  ): Function {
    const secretAccessStatement = new PolicyStatement({
      effect: Effect.ALLOW,
      actions: ["secretsmanager:GetSecretValue"],
      resources: [AwsConfig.OPENAI_API_SECRET_ARN],
    });

    const analysisSchedulingLambdaId = props.idBuilder.build("analysis-scheduling-lambda");
    const analysisSchedulingLambda = new Function(this, analysisSchedulingLambdaId, {
      ...lambdaCodeRunTimeConfig,
      ...this.buildLambdaLayersConfig([
        AWS_LAMBDA_LAYERS.AWS_QUEUE_LAMBDA_LAYER,
        AWS_LAMBDA_LAYERS.AWS_DYNAMODB_LAMBDA_LAYER,
      ]),
      functionName: analysisSchedulingLambdaId,
      description: "Lambda function to schedule analysis execution",
      handler: "AnalysisSchedulingLambda.handler",
      environment: {
        ANALYSIS_EXECUTIONS_QUEUE_URL: props.analysisExecutionQueue.queueUrl,
        ANALYSIS_EXECUTIONS_TABLE: props.analysisExecutionsTable.tableName,
      },
      timeout: AwsConfig.QUEUE_TIMEOUT_MINS,
      memorySize: 256,
    });

    analysisSchedulingLambda.addToRolePolicy(secretAccessStatement);
    props.analysisExecutionQueue.grantSendMessages(analysisSchedulingLambda);
    props.analysisExecutionsTable.grantReadWriteData(analysisSchedulingLambda);

    return analysisSchedulingLambda;
  }
}
