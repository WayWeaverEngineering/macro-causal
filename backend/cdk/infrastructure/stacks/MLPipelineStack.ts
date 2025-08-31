import { Construct } from 'constructs';
import { Stack, StackProps, Duration } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../../utils/Naming';
import { EcsFargateServiceConstruct } from '../constructs/EcsFargateServiceConstruct';
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

    const dataCollectionStageName = 'data-collection';
    const dataCollectionEcsId = DefaultIdBuilder.build(`${dataCollectionStageName}-ecs-service`);
    const dataCollectionEcs = new EcsFargateServiceConstruct(this, dataCollectionEcsId, {
      name: dataCollectionStageName,
      // IMPORTANT: the image path is relative to cdk.out
      imagePath: `../pipeline/${dataCollectionStageName}`,
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
    dataCollectionEcs.service.taskDefinition.taskRole.addToPrincipalPolicy(secretAccessStatement);

    // Enable data collection stage to write raw data to bronze bucket
    props.dataLakeStack.bronzeBucket.grantReadWrite(dataCollectionEcs.service.taskDefinition.taskRole);

    // Create Step Functions task for data collection stage
    const dataCollectionTaskId = DefaultIdBuilder.build(`${dataCollectionStageName}-task`);
    const dataCollectionTask = new tasks.EcsRunTask(this, dataCollectionTaskId, {
      stateName: dataCollectionStageName,
      cluster: dataCollectionEcs.service.cluster,
      taskDefinition: dataCollectionEcs.service.taskDefinition,
      integrationPattern: sfn.IntegrationPattern.RUN_JOB,
      
      // Configure task parameters
      launchTarget: new tasks.EcsFargateLaunchTarget({
        platformVersion: ecs.FargatePlatformVersion.LATEST,
      }),
      
      // Add container overrides with execution context
      containerOverrides: [{
        containerDefinition: dataCollectionEcs.service.taskDefinition.defaultContainer!,
        environment: [
          { name: 'EXECUTION_MODE', value: 'step-functions' },
          { name: 'PIPELINE_EXECUTION_ID', value: sfn.JsonPath.stringAt('$$.Execution.Id') },
          { name: 'EXECUTION_START_TIME', value: sfn.JsonPath.stringAt('$$.Execution.StartTime') }
        ]
      }],
      
      // Configure timeout
      taskTimeout: sfn.Timeout.duration(Duration.hours(2)), // 2-hour timeout for data collection
      
      // Result handling
      resultPath: '$.dataCollectionResult'
    });

    // Add validation for data collection success/failure
    const validateDataCollectionId = DefaultIdBuilder.build(`validate-${dataCollectionStageName}`);
    const validateDataCollection = new sfn
      .Choice(this, validateDataCollectionId, {
        stateName: `validate-${dataCollectionStageName}`
      })
      .when(sfn.Condition.stringEquals('$.dataCollectionResult.status', 'SUCCESS'), 
        new sfn.Succeed(this, `${dataCollectionStageName}-success`, {
          comment: 'Data collection completed successfully'
        }))
      .otherwise(new sfn.Fail(this, `${dataCollectionStageName}-failed`, {
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
      definitionBody: sfn.DefinitionBody.fromChainable(workflow),
      stateMachineName: stateMachineId,
      timeout: Duration.hours(24), // 24-hour timeout for entire pipeline
      comment: 'ML Pipeline for Macro Causal Analysis'
    });

    // In production, we would use a scheduled rule to trigger the pipeline
    // For now, we will manually trigger the pipeline from the AWS console
  }
}
