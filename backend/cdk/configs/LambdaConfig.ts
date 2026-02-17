export class LambdaConfig {
  // When the lambda code is built, the output artifact will be placed in backend/dist/lambda/src
  // We need to construct a path there that is relative to child directories of "stacks"
  static readonly LAMBDA_CODE_RELATIVE_PATH = '../../dist/lambda/src'
}