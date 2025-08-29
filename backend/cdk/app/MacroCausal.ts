import 'source-map-support/register';
import { App } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS } from '../utils/Constants';
import { CICDStack } from '../infrastructure/cicd/CICDStack';
import { DataLakeStack } from '../infrastructure/stacks/DataLakeStack';
import { MLTrainingStack } from '../infrastructure/stacks/MLTrainingStack';
import { InferenceStack } from '../infrastructure/stacks/InferenceStack';
import { MonitoringStack } from '../infrastructure/stacks/MonitoringStack';
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
  
  // CI/CD Stack (existing)
  const cicdStack = new CICDStack(app, DefaultIdBuilder.build('ci-cd-stack'), {
    prebuiltLambdaLayerArns,
    env: {
      account: process.env.CDK_DEFAULT_ACCOUNT,
      region: process.env.CDK_DEFAULT_REGION || 'us-east-1'
    }
  });

  // Data Lake Stack
  const dataLakeStack = new DataLakeStack(app, DefaultIdBuilder.build('data-lake-stack'), {
    environment: environment,
    env: {
      account: process.env.CDK_DEFAULT_ACCOUNT,
      region: process.env.CDK_DEFAULT_REGION || 'us-east-1'
    }
  });

  // ML Training Stack
  const mlTrainingStack = new MLTrainingStack(app, DefaultIdBuilder.build('ml-training-stack'), {
    environment: environment,
    dataLakeStack: dataLakeStack,
    env: {
      account: process.env.CDK_DEFAULT_ACCOUNT,
      region: process.env.CDK_DEFAULT_REGION || 'us-east-1'
    }
  });

  // Inference Stack
  const inferenceStack = new InferenceStack(app, DefaultIdBuilder.build('inference-stack'), {
    environment: environment,
    mlTrainingStack: mlTrainingStack,
    env: {
      account: process.env.CDK_DEFAULT_ACCOUNT,
      region: process.env.CDK_DEFAULT_REGION || 'us-east-1'
    }
  });

  // Monitoring Stack
  const monitoringStack = new MonitoringStack(app, DefaultIdBuilder.build('monitoring-stack'), {
    environment: environment,
    inferenceStack: inferenceStack,
    mlTrainingStack: mlTrainingStack,
    env: {
      account: process.env.CDK_DEFAULT_ACCOUNT,
      region: process.env.CDK_DEFAULT_REGION || 'us-east-1'
    }
  });

  // Add dependencies
  mlTrainingStack.addDependency(dataLakeStack);
  inferenceStack.addDependency(mlTrainingStack);
  monitoringStack.addDependency(inferenceStack);
  monitoringStack.addDependency(mlTrainingStack);

  app.synth();
}

main();