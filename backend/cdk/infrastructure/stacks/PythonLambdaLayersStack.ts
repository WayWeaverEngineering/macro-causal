import { Stack, StackProps } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as lambda from 'aws-cdk-lib/aws-lambda';
import { LambdaConfig } from "../configs/LambdaConfig";
import { DefaultIdBuilder } from "../../utils/Naming";

export class PythonLambdaLayersStack extends Stack {

  public readonly botoLambdaLayer: lambda.ILayerVersion;
  public readonly numpyLambdaLayer: lambda.ILayerVersion;
  public readonly pandasLambdaLayer: lambda.ILayerVersion;
  public readonly yfinanceLambdaLayer: lambda.ILayerVersion;
  public readonly requestsLambdaLayer: lambda.ILayerVersion;
  public readonly dateutilsLambdaLayer: lambda.ILayerVersion;

  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const botoLambdaLayerId = DefaultIdBuilder.build('boto-lambda-layer');
    this.botoLambdaLayer = lambda.LayerVersion.fromLayerVersionArn(this, botoLambdaLayerId, LambdaConfig.BOTO_LAMBDA_LAYER_ARN);

    const requestsLambdaLayerId = DefaultIdBuilder.build('requests-lambda-layer');
    this.requestsLambdaLayer = lambda.LayerVersion.fromLayerVersionArn(this, requestsLambdaLayerId, LambdaConfig.REQUESTS_LAMBDA_LAYER_ARN);

    const dateutilsLambdaLayerId = DefaultIdBuilder.build('dateutils-lambda-layer');
    this.dateutilsLambdaLayer = lambda.LayerVersion.fromLayerVersionArn(this, dateutilsLambdaLayerId, LambdaConfig.DATEUTILS_LAMBDA_LAYER_ARN);

    const numpyLambdaLayerId = DefaultIdBuilder.build('numpy-lambda-layer');
    this.numpyLambdaLayer = lambda.LayerVersion.fromLayerVersionArn(this, numpyLambdaLayerId, LambdaConfig.NUMPY_LAMBDA_LAYER_ARN);

    const pandasLambdaLayerId = DefaultIdBuilder.build('pandas-lambda-layer');
    this.pandasLambdaLayer = lambda.LayerVersion.fromLayerVersionArn(this, pandasLambdaLayerId, LambdaConfig.PANDAS_LAMBDA_LAYER_ARN);

    const yfinanceLambdaLayerId = DefaultIdBuilder.build('yfinance-lambda-layer');
    this.yfinanceLambdaLayer = lambda.LayerVersion.fromLayerVersionArn(this, yfinanceLambdaLayerId, LambdaConfig.YFINANCE_LAMBDA_LAYER_ARN);
  }
}