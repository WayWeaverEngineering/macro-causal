import { Runtime } from 'aws-cdk-lib/aws-lambda';

export class LambdaConfig {

  static readonly DEFAULT_PYTHON_RUNTIME = Runtime.PYTHON_3_10;

  static readonly LAMBDA_CODE_FOLDER = "../lambda";

  static readonly LAMBDA_PYTHON_CODE_FOLDER = `${LambdaConfig.LAMBDA_CODE_FOLDER}/python`;

  static getLambdaPythonCodePath(subPath: string) {
    return `${LambdaConfig.LAMBDA_PYTHON_CODE_FOLDER}/${subPath}`;
  }

  static readonly BOTO_LAMBDA_LAYER_ARN = "arn:aws:lambda:us-east-1:715067592333:layer:macro-causal-boto-lambda-layer:12";
  static readonly NUMPY_LAMBDA_LAYER_ARN = "arn:aws:lambda:us-east-1:715067592333:layer:macro-causal-numpy-lambda-layer:12";
  static readonly PANDAS_LAMBDA_LAYER_ARN = "arn:aws:lambda:us-east-1:715067592333:layer:macro-causal-pandas-lambda-layer:12";
  static readonly REQUESTS_LAMBDA_LAYER_ARN = "arn:aws:lambda:us-east-1:715067592333:layer:macro-causal-requests-lambda-layer:8";
  static readonly YFINANCE_LAMBDA_LAYER_ARN = "arn:aws:lambda:us-east-1:715067592333:layer:macro-causal-yfinance-lambda-layer:12";
  static readonly DATEUTILS_LAMBDA_LAYER_ARN = "arn:aws:lambda:us-east-1:715067592333:layer:macro-causal-dateutils-lambda-layer:8";
}