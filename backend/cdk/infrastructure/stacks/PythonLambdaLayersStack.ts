import { Stack, StackProps } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as lambda from 'aws-cdk-lib/aws-lambda';
import { LambdaConfig } from "../configs/LambdaConfig";
import { DefaultIdBuilder } from "../../utils/Naming";

export class PythonLambdaLayersStack extends Stack {

  public readonly botoLambdaLayer: lambda.LayerVersion;
  public readonly numpyLambdaLayer: lambda.LayerVersion;
  public readonly pandasLambdaLayer: lambda.LayerVersion;
  public readonly yfinanceLambdaLayer: lambda.LayerVersion;
  public readonly requestsLambdaLayer: lambda.LayerVersion;
  public readonly dateutilsLambdaLayer: lambda.LayerVersion;

  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const layerBundlingConfig = {
      bundling: {
        image: lambda.Runtime.PYTHON_3_10.bundlingImage, // Amazon Linux + Python 3.10
        command: [
          'bash', '-c',
          [
            'pip install -r requirements.txt -t python', // install into ./python
            'cp -r . /asset-output'                     // copy into final asset
          ].join(' && ')
        ],
      },
    }

    const botoLambdaLayerId = DefaultIdBuilder.build('boto-lambda-layer');
    this.botoLambdaLayer = new lambda.LayerVersion(this, botoLambdaLayerId, {
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonLayerPath('boto'), layerBundlingConfig),
      compatibleRuntimes: [LambdaConfig.DEFAULT_PYTHON_RUNTIME],
      description: 'Python lambda layer for boto',
      layerVersionName: botoLambdaLayerId
    });

    const requestsLambdaLayerId = DefaultIdBuilder.build('requests-lambda-layer');
    this.requestsLambdaLayer = new lambda.LayerVersion(this, requestsLambdaLayerId, {
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonLayerPath('requests'), layerBundlingConfig),
      compatibleRuntimes: [LambdaConfig.DEFAULT_PYTHON_RUNTIME],
      description: 'Python lambda layer for requests',
      layerVersionName: requestsLambdaLayerId
    });

    const dateutilsLambdaLayerId = DefaultIdBuilder.build('dateutils-lambda-layer');
    this.dateutilsLambdaLayer = new lambda.LayerVersion(this, dateutilsLambdaLayerId, {
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonLayerPath('dateutils'), layerBundlingConfig),
      compatibleRuntimes: [LambdaConfig.DEFAULT_PYTHON_RUNTIME],
      description: 'Python lambda layer for dateutils',
      layerVersionName: dateutilsLambdaLayerId
    });

    const numpyLambdaLayerId = DefaultIdBuilder.build('numpy-lambda-layer');
    this.numpyLambdaLayer = new lambda.LayerVersion(this, numpyLambdaLayerId, {
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonLayerPath('numpy'), layerBundlingConfig),
      compatibleRuntimes: [LambdaConfig.DEFAULT_PYTHON_RUNTIME],
      description: 'Python lambda layer for numpy',
      layerVersionName: numpyLambdaLayerId
    });

    const pandasLambdaLayerId = DefaultIdBuilder.build('pandas-lambda-layer');
    this.pandasLambdaLayer = new lambda.LayerVersion(this, pandasLambdaLayerId, {
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonLayerPath('pandas'), layerBundlingConfig),
      compatibleRuntimes: [LambdaConfig.DEFAULT_PYTHON_RUNTIME],
      description: 'Python lambda layer for pandas',
      layerVersionName: pandasLambdaLayerId
    });

    const yfinanceLambdaLayerId = DefaultIdBuilder.build('yfinance-lambda-layer');
    this.yfinanceLambdaLayer = new lambda.LayerVersion(this, yfinanceLambdaLayerId, {
      code: lambda.Code.fromAsset(LambdaConfig.getLambdaPythonLayerPath('yfinance'), layerBundlingConfig),
      compatibleRuntimes: [LambdaConfig.DEFAULT_PYTHON_RUNTIME],
      description: 'Python lambda layer for yfinance',
      layerVersionName: yfinanceLambdaLayerId
    });
  }
}