import { Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { RESOURCE_NAMES } from '../../utils/Constants';
import { InferenceConstruct } from '../constructs/InferenceConstruct';

export interface InferenceStackProps extends StackProps {
  environment: string;
  mlTrainingStack: any; // MLTrainingStack
}

export class InferenceStack extends Stack {
  public readonly inference: InferenceConstruct;

  constructor(scope: Construct, id: string, props: InferenceStackProps) {
    super(scope, id, props);

    // Inference construct
    this.inference = new InferenceConstruct(this, RESOURCE_NAMES.INFERENCE_CONSTRUCT, {
      environment: props.environment,
      accountId: this.account,
      region: this.region,
      artifactsBucket: props.mlTrainingStack.modelSaving.artifactsBucket,
      registryTable: props.mlTrainingStack.modelSaving.registryTable
    });
  }
}
