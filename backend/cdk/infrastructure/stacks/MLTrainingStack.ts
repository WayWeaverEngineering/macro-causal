import { Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { MLTrainingConstruct } from '../constructs/MLTrainingConstruct';
import { ModelSavingConstruct } from '../constructs/ModelSavingConstruct';
import { DefaultIdBuilder } from '../../utils/Naming';

export interface MLTrainingStackProps extends StackProps {
  dataLakeStack: any; // DataLakeStack
}

export class MLTrainingStack extends Stack {
  public readonly mlTraining: MLTrainingConstruct;
  public readonly modelSaving: ModelSavingConstruct;

  constructor(scope: Construct, id: string, props: MLTrainingStackProps) {
    super(scope, id, props);

    // ML Training construct
    const mlTrainingId = DefaultIdBuilder.build('ml-training');
    this.mlTraining = new MLTrainingConstruct(this, mlTrainingId, {
      accountId: this.account,
      region: this.region,
      goldBucket: props.dataLakeStack.dataLake.goldBucket,
      artifactsBucket: props.dataLakeStack.dataLake.artifactsBucket, // This will be created in ModelSavingConstruct
    });

    // Model Saving construct
    const modelSavingId = DefaultIdBuilder.build('model-saving');
    this.modelSaving = new ModelSavingConstruct(this, modelSavingId, {
      accountId: this.account,
      region: this.region,
      trainingRole: this.mlTraining.trainingRole,
    });

    // Grant access to gold bucket for training
    props.dataLakeStack.dataLake.goldBucket.grantRead(this.mlTraining.trainingRole);

    // Grant access to artifacts bucket for training
    this.modelSaving.artifactsBucket.grantReadWrite(this.mlTraining.trainingRole);
  }
}
