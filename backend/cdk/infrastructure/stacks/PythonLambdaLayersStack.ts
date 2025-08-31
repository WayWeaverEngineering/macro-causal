import { Stack, StackProps, RemovalPolicy } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as lambda from 'aws-cdk-lib/aws-lambda';
import { LambdaConfig } from "../configs/LambdaConfig";
import { DefaultIdBuilder } from "../../utils/Naming";

export class PythonLambdaLayersStack extends Stack {

  public readonly botoLambdaLayer: lambda.LayerVersion;
  public readonly numpyLambdaLayer: lambda.LayerVersion;
  public readonly pandasLambdaLayer: lambda.LayerVersion;
  public readonly requestsLambdaLayer: lambda.LayerVersion;
  public readonly dateutilsLambdaLayer: lambda.LayerVersion;

  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const botoLambdaLayerId = DefaultIdBuilder.build('boto-lambda-layer');
    this.botoLambdaLayer = new lambda.LayerVersion(this, botoLambdaLayerId, {
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonLayerPath('boto-layer')),
      compatibleRuntimes: [LambdaConfig.DEFAULT_PYTHON_RUNTIME],
      description: 'Python lambda layer for boto',
      layerVersionName: botoLambdaLayerId,
      removalPolicy: RemovalPolicy.RETAIN
    });

    const requestsLambdaLayerId = DefaultIdBuilder.build('requests-lambda-layer');
    this.requestsLambdaLayer = new lambda.LayerVersion(this, requestsLambdaLayerId, {
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonLayerPath('requests-layer')),
      compatibleRuntimes: [LambdaConfig.DEFAULT_PYTHON_RUNTIME],
      description: 'Python lambda layer for requests',
      layerVersionName: requestsLambdaLayerId,
      removalPolicy: RemovalPolicy.RETAIN
    });

    const dateutilsLambdaLayerId = DefaultIdBuilder.build('dateutils-lambda-layer');
    this.dateutilsLambdaLayer = new lambda.LayerVersion(this, dateutilsLambdaLayerId, {
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonLayerPath('dateutils-layer')),
      compatibleRuntimes: [LambdaConfig.DEFAULT_PYTHON_RUNTIME],
      description: 'Python lambda layer for dateutils',
      layerVersionName: dateutilsLambdaLayerId,
      removalPolicy: RemovalPolicy.RETAIN
    });

    const numpyLambdaLayerId = DefaultIdBuilder.build('numpy-lambda-layer');
    this.numpyLambdaLayer = new lambda.LayerVersion(this, numpyLambdaLayerId, {
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonLayerPath('numpy-layer')),
      compatibleRuntimes: [LambdaConfig.DEFAULT_PYTHON_RUNTIME],
      description: 'Python lambda layer for numpy',
      layerVersionName: numpyLambdaLayerId,
      removalPolicy: RemovalPolicy.RETAIN
    });

    const pandasLambdaLayerId = DefaultIdBuilder.build('pandas-lambda-layer');
    this.pandasLambdaLayer = new lambda.LayerVersion(this, pandasLambdaLayerId, {
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonLayerPath('pandas-layer')),
      compatibleRuntimes: [LambdaConfig.DEFAULT_PYTHON_RUNTIME],
      description: 'Python lambda layer for pandas',
      layerVersionName: pandasLambdaLayerId,
      removalPolicy: RemovalPolicy.RETAIN
    });
  }
}