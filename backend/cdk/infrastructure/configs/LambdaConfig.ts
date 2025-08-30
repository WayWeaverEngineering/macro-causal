import { Runtime } from 'aws-cdk-lib/aws-lambda';

export class LambdaConfig {

  static readonly DEFAULT_PYTHON_RUNTIME = Runtime.PYTHON_3_10;

  static readonly LAMBDA_CODE_FOLDER = "../lambda";

  static readonly LAMBDA_PYTHON_CODE_FOLDER = `${LambdaConfig.LAMBDA_CODE_FOLDER}/python`;

  static getLambdaPythonCodePath(subPath: string) {
    return `${LambdaConfig.LAMBDA_PYTHON_CODE_FOLDER}/${subPath}`;
  }

  static getLambdaPythonLayerPath(subPath: string) {
    return `${LambdaConfig.LAMBDA_PYTHON_CODE_FOLDER}/layers/${subPath}`;
  }
}