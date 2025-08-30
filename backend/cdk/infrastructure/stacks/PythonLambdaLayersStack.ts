import { Stack, StackProps } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as lambda from 'aws-cdk-lib/aws-lambda';
import { LambdaConfig } from "../configs/LambdaConfig";
import { DefaultIdBuilder } from "../../utils/Naming";

export class PythonLambdaLayersStack extends Stack {

  public readonly botoLambdaLayer: lambda.LayerVersion;
  public readonly utilsLambdaLayer: lambda.LayerVersion;
  public readonly numpyLambdaLayer: lambda.LayerVersion;
  public readonly pandasLambdaLayer: lambda.LayerVersion;
  public readonly yfinanceLambdaLayer: lambda.LayerVersion;

  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const botoLambdaLayerId = DefaultIdBuilder.build('boto-lambda-layer');
    this.botoLambdaLayer = new lambda.LayerVersion(this, botoLambdaLayerId, {
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonLayerPath('boto')),
      compatibleRuntimes: [LambdaConfig.DEFAULT_PYTHON_RUNTIME],
      description: 'Python lambda layer for boto',
      layerVersionName: botoLambdaLayerId
    });

    const utilsLambdaLayerId = DefaultIdBuilder.build('utils-lambda-layer');
    this.utilsLambdaLayer = new lambda.LayerVersion(this, utilsLambdaLayerId, {
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonLayerPath('utils')),
      compatibleRuntimes: [LambdaConfig.DEFAULT_PYTHON_RUNTIME],
      description: 'Python lambda layer for common utils',
      layerVersionName: utilsLambdaLayerId
    });

    const numpyLambdaLayerId = DefaultIdBuilder.build('numpy-lambda-layer');
    this.numpyLambdaLayer = new lambda.LayerVersion(this, numpyLambdaLayerId, {
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonLayerPath('numpy')),
      compatibleRuntimes: [LambdaConfig.DEFAULT_PYTHON_RUNTIME],
      description: 'Python lambda layer for numpy',
      layerVersionName: numpyLambdaLayerId
    });

    const pandasLambdaLayerId = DefaultIdBuilder.build('pandas-lambda-layer');
    this.pandasLambdaLayer = new lambda.LayerVersion(this, pandasLambdaLayerId, {
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonLayerPath('pandas')),
      compatibleRuntimes: [LambdaConfig.DEFAULT_PYTHON_RUNTIME],
      description: 'Python lambda layer for pandas',
      layerVersionName: pandasLambdaLayerId
    });

    const yfinanceLambdaLayerId = DefaultIdBuilder.build('yfinance-lambda-layer');
    this.yfinanceLambdaLayer = new lambda.LayerVersion(this, yfinanceLambdaLayerId, {
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonLayerPath('yfinance')),
      compatibleRuntimes: [LambdaConfig.DEFAULT_PYTHON_RUNTIME],
      description: 'Python lambda layer for yfinance',
      layerVersionName: yfinanceLambdaLayerId
    });
  }
}