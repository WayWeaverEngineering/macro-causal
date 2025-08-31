import { Construct } from 'constructs';
import { Stack, StackProps, Duration } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../../utils/Naming';
import { PipelineStageConstruct } from '../constructs/PipelineStageConstruct';
import { DataLakeStack } from './DataLakeStack';
import { AwsConfig } from '../configs/AwsConfig';
import { Effect, PolicyStatement } from 'aws-cdk-lib/aws-iam';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as ecs from 'aws-cdk-lib/aws-ecs';

export interface MLPipelineStackProps extends StackProps {
  dataLakeStack: DataLakeStack;
}

export class MLPipelineStack extends Stack {
  public readonly pipelineStateMachine: sfn.StateMachine;

  constructor(scope: Construct, id: string, props: MLPipelineStackProps) {
    super(scope, id, props);

    const dataCollectionStage = new PipelineStageConstruct(
      this, DefaultIdBuilder.build('data-collection-stage'), {
      stageName: 'data-collection',
      environment: {
        BRONZE_BUCKET: props.dataLakeStack.bronzeBucket.bucketName,
        API_SECRETS_ARN: AwsConfig.FRED_API_SECRET_ARN
      }
    });

    // IAM policy statement to allow pipeline stages to fetch secrets from Secret Manager
    const secretAccessStatement = new PolicyStatement({
      effect: Effect.ALLOW,
      actions: ['secretsmanager:GetSecretValue'],
      resources: [
        AwsConfig.FRED_API_SECRET_ARN
      ]
    })

    // Add secret access statement to data collection stage task role
    dataCollectionStage.service.taskDefinition.taskRole.addToPrincipalPolicy(secretAccessStatement);

    // Enable data collection stage to write raw data to bronze bucket
    props.dataLakeStack.bronzeBucket.grantReadWrite(dataCollectionStage.service.taskDefinition.taskRole);

    // Create Step Functions task for data collection stage
    const dataCollectionTaskId = DefaultIdBuilder.build('data-collection-task');
    const dataCollectionTask = new tasks.EcsRunTask(this, dataCollectionTaskId, {
      stateName: "DataCollection",
      cluster: dataCollectionStage.service.cluster,
      taskDefinition: dataCollectionStage.service.taskDefinition,
      integrationPattern: sfn.IntegrationPattern.RUN_JOB,
      
      // Configure task parameters
      launchTarget: new tasks.EcsFargateLaunchTarget({
        platformVersion: ecs.FargatePlatformVersion.LATEST,
      }),
      
      // Add container overrides with execution context
      containerOverrides: [{
        containerDefinition: dataCollectionStage.service.taskDefinition.defaultContainer!,
        environment: [
          { name: 'EXECUTION_MODE', value: 'step-functions' },
          { name: 'PIPELINE_EXECUTION_ID', value: sfn.JsonPath.stringAt('$$.Execution.Id') },
          { name: 'EXECUTION_START_TIME', value: sfn.JsonPath.stringAt('$$.Execution.StartTime') }
        ]
      }],
      
      // Configure timeout
      timeout: Duration.hours(2), // 2-hour timeout for data collection
      
      // Result handling
      resultPath: '$.dataCollectionResult'
    });

    // Add validation for data collection success/failure
    const validateDataCollectionId = DefaultIdBuilder.build('validate-data-collection');
    const validateDataCollection = new sfn
      .Choice(this, validateDataCollectionId, {
        stateName: "ValidateDataCollection"
      })
      .when(sfn.Condition.stringEquals('$.dataCollectionResult.status', 'SUCCESS'), 
        new sfn.Succeed(this, 'DataCollectionSuccess', {
          comment: 'Data collection completed successfully'
        }))
      .otherwise(new sfn.Fail(this, 'DataCollectionFailed', {
        error: 'DataCollectionFailed',
        cause: 'Data collection stage failed',
        comment: 'Data collection stage encountered an error'
      }));

    // Build the workflow
    const workflow = sfn.Chain
      .start(dataCollectionTask)
      .next(validateDataCollection);

    // Create the state machine
    const stateMachineId = DefaultIdBuilder.build('ml-pipeline-state-machine');
    this.pipelineStateMachine = new sfn.StateMachine(this, stateMachineId, {
      definition: workflow,
      stateMachineName: stateMachineId,
      timeout: Duration.hours(24), // 24-hour timeout for entire pipeline
      comment: 'ML Pipeline for Macro Causal Analysis'
    });

    // In production, we would use a scheduled rule to trigger the pipeline
    // For now, we will manually trigger the pipeline from the AWS console
  }
}
