import 'source-map-support/register';
import { App } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS } from '../utils/Constants';
import { CICDStack } from '../infrastructure/cicd/CICDStack';
import {
  COMMON_UTILS_LAMBDA_LAYER_NAME,
  PrebuiltLambdaLayersStack,
  SsmParamClient
} from '@wayweaver/ariadne';

async function main() {
  const app = new App();
  
  // Get environment from context or default to dev
  const environment = app.node.tryGetContext('environment') || MACRO_CAUSAL_CONSTANTS.ENVIRONMENTS.DEV;
  
  // Bootstrap pre-built Lambda layer ARNs from SSM
  const ssmClient = new SsmParamClient({ isCI: true });
  const layerNames = [
    COMMON_UTILS_LAMBDA_LAYER_NAME
  ];
  const prebuiltLambdaLayerArns = await PrebuiltLambdaLayersStack.getArnsfromLayerNames(layerNames, ssmClient);

  new CICDStack(app, DefaultIdBuilder.build('ci-cd-stack'), {
    prebuiltLambdaLayerArns,
    environment
  });

  app.synth();
}

main();