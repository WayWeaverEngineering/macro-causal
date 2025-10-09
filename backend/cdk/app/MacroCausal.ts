import 'source-map-support/register';
import { App } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../utils/Naming';
import { CICDStack } from '../infrastructure/cicd/CICDStack';
import {
  AWS_QUEUE_LAMBDA_LAYER_NAME,
  LANGCHAIN_LANGGRAPH_LAMBDA_LAYER_NAME,
  AWS_DYNAMODB_LAMBDA_LAYER_NAME,
  AWS_OPENSEARCH_LAMBDA_LAYER_NAME,
  AWS_EMR_SERVERLESS_LAMBDA_LAYER_NAME,
  AWS_ECS_LAMBDA_LAYER_NAME,
  COMMON_UTILS_LAMBDA_LAYER_NAME,
  PrebuiltLambdaLayersStack,
  SsmParamClient
} from '@wayweaver/ariadne';

async function main() {
  const app = new App();
  
  // Bootstrap pre-built Lambda layer ARNs from SSM
  const ssmClient = new SsmParamClient({ isCI: true });
  const layerNames = [
    COMMON_UTILS_LAMBDA_LAYER_NAME,
    AWS_ECS_LAMBDA_LAYER_NAME,
    AWS_EMR_SERVERLESS_LAMBDA_LAYER_NAME,
    LANGCHAIN_LANGGRAPH_LAMBDA_LAYER_NAME,
    AWS_DYNAMODB_LAMBDA_LAYER_NAME,
    AWS_OPENSEARCH_LAMBDA_LAYER_NAME,
    AWS_QUEUE_LAMBDA_LAYER_NAME
  ];
  const prebuiltLambdaLayerArns = await PrebuiltLambdaLayersStack.getArnsfromLayerNames(layerNames, ssmClient);

  new CICDStack(app, DefaultIdBuilder.build('ci-cd-stack'), {
    prebuiltLambdaLayerArns
  });

  app.synth();
}

main();