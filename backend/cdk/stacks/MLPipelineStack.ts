import { Construct } from 'constructs';
import { Stack, StackProps, Duration } from 'aws-cdk-lib';
import { ConstructIdBuilder, PrebuiltLambdaLayers } from '@wayweaver/ariadne';
import { DataLakeStack } from './DataLakeStack';
import { DataCollectionStage } from '../stages/DataCollectionStage';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import { DataProcessingStage } from '../stages/DataProcessingStage';
import { ModelTrainingStage } from '../stages/ModelTrainingStage';
import { ModelServingStage } from '../stages/ModelServingStage';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';

export interface MLPipelineStackProps extends StackProps {
  idBuilder: ConstructIdBuilder;
  dataLakeStack: DataLakeStack;
  prebuiltLambdaLayers: PrebuiltLambdaLayers;
  modelRegistryTable: dynamodb.Table;
}

export class MLPipelineStack extends Stack {
  public readonly pipelineStateMachine: sfn.StateMachine;

  constructor(scope: Construct, id: string, props: MLPipelineStackProps) {
    super(scope, id, props);

    const dataCollectionStageId = props.idBuilder.build('data-collection-stage');
    const dataCollectionStage = new DataCollectionStage(this, dataCollectionStageId, {
      idBuilder: props.idBuilder,
      dataLakeStack: props.dataLakeStack
    });
    
    const dataProcessingStageId = props.idBuilder.build('data-processing-stage');
    const dataProcessingStage = new DataProcessingStage(this, dataProcessingStageId, {
      idBuilder: props.idBuilder,
      dataLakeStack: props.dataLakeStack,
      prebuiltLambdaLayers: props.prebuiltLambdaLayers
    });

    const modelTrainingStageId = props.idBuilder.build('model-training-stage');
    const modelTrainingStage = new ModelTrainingStage(this, modelTrainingStageId, {
      idBuilder: props.idBuilder,
      dataLakeStack: props.dataLakeStack,
      prebuiltLambdaLayers: props.prebuiltLambdaLayers,
      modelRegistryTable: props.modelRegistryTable
    });

    const modelServingStageId = props.idBuilder.build('model-serving-stage');
    const modelServingStage = new ModelServingStage(this, modelServingStageId, {
      idBuilder: props.idBuilder,
      dataLakeStack: props.dataLakeStack,
      modelRegistryTable: props.modelRegistryTable
    });

    // Chain all stages together
    const mlWorkflow = sfn.Chain
      .start(dataCollectionStage)
      .next(dataProcessingStage)
      .next(modelTrainingStage)
      .next(modelServingStage);

    // Create the state machine
    const stateMachineId = props.idBuilder.build('ml-pipeline-state-machine');
    this.pipelineStateMachine = new sfn.StateMachine(this, stateMachineId, {
      definitionBody: sfn.DefinitionBody.fromChainable(mlWorkflow),
      stateMachineName: stateMachineId,
      timeout: Duration.hours(24), // 24-hour timeout for entire pipeline
      comment: 'ML Pipeline for Macro Causal Analysis with Ray Training'
    });

    // In production, we would use a scheduled rule to trigger the pipeline
    // For now, we will manually trigger the pipeline from the AWS console
  }
}
