import 'source-map-support/register';
import { App } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../utils/Naming';
import { CICDStack } from '../infrastructure/cicd/CICDStack';
import {
  COMMON_UTILS_LAMBDA_LAYER_NAME,
  PrebuiltLambdaLayersStack,
  SsmParamClient
} from '@wayweaver/ariadne';

async function main() {
  const ssmClient = new SsmParamClient({ isCI: true });

  // Bootstrap pre-built Lambda layer ARNs from SSM
  const layerNames = [
    COMMON_UTILS_LAMBDA_LAYER_NAME
  ];
  const prebuiltLambdaLayerArns = await PrebuiltLambdaLayersStack.getArnsfromLayerNames(layerNames, ssmClient);

  const app = new App();
  new CICDStack(app, DefaultIdBuilder.build('ci-cd-stack'), {
    prebuiltLambdaLayerArns
  });
  app.synth();
}

main();