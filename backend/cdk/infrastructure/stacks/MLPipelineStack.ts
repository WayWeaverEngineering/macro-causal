import { Construct } from 'constructs';
import { Stack, StackProps } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../../utils/Naming';
import { PipelineStageConstruct } from '../constructs/PipelineStageConstruct';
import { DataLakeStack } from './DataLakeStack';

export interface MLPipelineStackProps extends StackProps {
  dataLakeStack: DataLakeStack;
}

export class MLPipelineStack extends Stack {

  constructor(scope: Construct, id: string, props: MLPipelineStackProps) {
    super(scope, id, props);

    const dataCollectionStage = new PipelineStageConstruct(
      this, DefaultIdBuilder.build('data-collection-stage'), {
      stageName: 'data-collection',
      environment: {
        BRONZE_BUCKET: props.dataLakeStack.bronzeBucket.bucketName,
      }
    });

    // Enable data collection stage to write raw data to bronze bucket
    props.dataLakeStack.bronzeBucket.grantReadWrite(dataCollectionStage.service.taskDefinition.taskRole);
  }
}
