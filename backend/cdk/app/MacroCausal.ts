import 'source-map-support/register';
import { App } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../utils/Naming';
import { CICDStack } from '../infrastructure/cicd/CICDStack';

async function main() {
  const app = new App();
  new CICDStack(app, DefaultIdBuilder.build('ci-cd-stack'));
  app.synth();
}

main();