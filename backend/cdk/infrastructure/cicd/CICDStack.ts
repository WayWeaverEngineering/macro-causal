import { Construct } from 'constructs';
import { Stack, StackProps } from 'aws-cdk-lib';
import { CodeBuildPipeline } from './CodeBuildPipeline';
import { DeploymentStage } from './DeploymentStage';
import { DefaultIdBuilder } from '../../utils/Naming';
import { LayerArns } from '@wayweaver/ariadne';

interface CICDStackProps extends StackProps {
  prebuiltLambdaLayerArns: LayerArns
}

export class CICDStack extends Stack {
  constructor(scope: Construct, id: string, props: CICDStackProps) {
    super(scope, id, props);

    // Create pipeline to build the code
    const pipelineId = DefaultIdBuilder.build('code-build-pipeline');
    const codeBuildPipeline = new CodeBuildPipeline(this, pipelineId);

    // Add deployment stage to deploy the built code
    const stageId = DefaultIdBuilder.build('deployment-stage');
    codeBuildPipeline.addStage(new DeploymentStage(this, stageId, {
      prebuiltLambdaLayerArns: props.prebuiltLambdaLayerArns
    }));
  }
}
