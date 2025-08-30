import { Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { InferenceConstruct } from '../constructs/InferenceConstruct';
import { VPCStack } from './VPCStack';
import { DefaultIdBuilder } from '../../utils/Naming';

export interface InferenceStackProps extends StackProps {
  mlTrainingStack: any; // MLTrainingStack
  vpcStack: VPCStack;
}

export class InferenceStack extends Stack {
  public readonly inference: InferenceConstruct;

  constructor(scope: Construct, id: string, props: InferenceStackProps) {
    super(scope, id, props);

    // Inference construct
    const inferenceId = DefaultIdBuilder.build('inference');
    this.inference = new InferenceConstruct(this, inferenceId, {
      vpc: props.vpcStack.vpcConstruct.vpc,
      accountId: this.account,
      region: this.region,
      artifactsBucket: props.mlTrainingStack.modelSaving.artifactsBucket,
      registryTable: props.mlTrainingStack.modelSaving.registryTable
    });
  }
}
