import { Construct } from "constructs";
import { AwsConfig } from "../configs/AwsConfig";
import { Effect, PolicyStatement } from "aws-cdk-lib/aws-iam";
import { DefaultIdBuilder } from "../../utils/Naming";
import { EcsFargateServiceConstruct } from "./EcsFargateServiceConstruct";
import { DataLakeStack } from "../stacks/DataLakeStack";
import { Duration } from "aws-cdk-lib";
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as ecs from 'aws-cdk-lib/aws-ecs';

export interface DataCollectionStageProps {
  dataLakeStack: DataLakeStack;
}

export class DataCollectionStage extends Construct {

  readonly workflow: sfn.Chain;

  constructor(scope: Construct, id: string, props: DataCollectionStageProps) {
    super(scope, id);

    const dataCollectionStageName = 'data-collection';
    const dataCollectionEcsId = DefaultIdBuilder.build(`${dataCollectionStageName}-ecs`);
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

    const dataCollectionTaskId = DefaultIdBuilder.build(`${dataCollectionStageName}-task`);
    const dataCollectionTask = new tasks.EcsRunTask(this, dataCollectionTaskId, {
      stateName: dataCollectionStageName,
      comment: "Data collection using Dockerized Python application on ECS Fargate",
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

    this.workflow = sfn.Chain.start(dataCollectionTask)
  }
}