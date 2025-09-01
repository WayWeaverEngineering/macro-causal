import { Construct } from 'constructs';
import { Stack, StackProps, Duration } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../../utils/Naming';
import { DataLakeStack } from './DataLakeStack';
import { DataCollectionStage } from '../stages/DataCollectionStage';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import { DataProcessingStage } from '../stages/DataProcessingStage';
import { ModelTrainingStage } from '../stages/ModelTrainingStage';
import { AWS_CLIENT_ECS_LAMBDA_LAYER_NAME, AWS_CLIENT_EMR_SERVERLESS_LAMBDA_LAYER_NAME, COMMON_UTILS_LAMBDA_LAYER_NAME, PrebuiltLambdaLayersStack } from '@wayweaver/ariadne';
import { ModelServingStage } from '../stages/ModelServingStage';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';

export interface MLPipelineStackProps extends StackProps {
  dataLakeStack: DataLakeStack;
  lambdaLayersStack: PrebuiltLambdaLayersStack;
  modelRegistryTable: dynamodb.Table;
}

export class MLPipelineStack extends Stack {
  public readonly pipelineStateMachine: sfn.StateMachine;

  constructor(scope: Construct, id: string, props: MLPipelineStackProps) {
    super(scope, id, props);

    const dataCollectionStageId = DefaultIdBuilder.build('data-collection-stage');
    const dataCollectionStage = new DataCollectionStage(this, dataCollectionStageId, {
      dataLakeStack: props.dataLakeStack
    });

    const commonUtilsLambdaLayer = props.lambdaLayersStack.getLayer(COMMON_UTILS_LAMBDA_LAYER_NAME)
    const emrServerlessLambdaLayer = props.lambdaLayersStack.getLayer(AWS_CLIENT_EMR_SERVERLESS_LAMBDA_LAYER_NAME)
    const ecsLambdaLayer = props.lambdaLayersStack.getLayer(AWS_CLIENT_ECS_LAMBDA_LAYER_NAME)
    
    const dataProcessingStageId = DefaultIdBuilder.build('data-processing-stage');
    const dataProcessingStage = new DataProcessingStage(this, dataProcessingStageId, {
      dataLakeStack: props.dataLakeStack,
      commonUtilsLambdaLayer,
      emrServerlessLambdaLayer
    });

    const modelTrainingStageId = DefaultIdBuilder.build('model-training-stage');
    const modelTrainingStage = new ModelTrainingStage(this, modelTrainingStageId, {
      dataLakeStack: props.dataLakeStack,
      commonUtilsLambdaLayer,
      ecsLambdaLayer,
      modelRegistryTable: props.modelRegistryTable
    });

    const modelServingStageId = DefaultIdBuilder.build('model-serving-stage');
    const modelServingStage = new ModelServingStage(this, modelServingStageId, {
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
    const stateMachineId = DefaultIdBuilder.build('ml-pipeline-state-machine');
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
