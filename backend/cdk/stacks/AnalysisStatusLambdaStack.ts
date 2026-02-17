import { Construct } from "constructs";
import { Function } from "aws-cdk-lib/aws-lambda";
import {
  AWS_LAMBDA_LAYERS,
  BaseLambdaStack,
  BaseLambdaStackProps,
  LambdaCodeRunTimeConfig,
} from "@wayweaver/ariadne";
import { Table } from "aws-cdk-lib/aws-dynamodb";
import { AwsConfig } from "../configs/AwsConfig";

export interface AnalysisStatusLambdaStackProps extends BaseLambdaStackProps {
  analysisExecutionsTable: Table;
}

export class AnalysisStatusLambdaStack extends BaseLambdaStack {
  constructor(scope: Construct, id: string, props: AnalysisStatusLambdaStackProps) {
    super(scope, id, props);
  }

  protected buildLambdaFunction(
    props: AnalysisStatusLambdaStackProps,
    lambdaCodeRunTimeConfig: LambdaCodeRunTimeConfig
  ): Function {
    const analysisStatusLambdaId = props.idBuilder.build("analysis-status-lambda");
    const analysisStatusLambda = new Function(this, analysisStatusLambdaId, {
      ...lambdaCodeRunTimeConfig,
      ...this.buildLambdaLayersConfig([
        AWS_LAMBDA_LAYERS.AWS_DYNAMODB_LAMBDA_LAYER,
      ]),
      functionName: analysisStatusLambdaId,
      description: "Lambda function to get analysis status",
      handler: "AnalysisStatusLambda.handler",
      environment: {
        ANALYSIS_EXECUTIONS_TABLE: props.analysisExecutionsTable.tableName,
      },
      timeout: AwsConfig.QUEUE_TIMEOUT_MINS,
      memorySize: 256,
    });

    props.analysisExecutionsTable.grantReadData(analysisStatusLambda);

    return analysisStatusLambda;
  }
}
