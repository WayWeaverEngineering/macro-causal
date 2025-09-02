import { Construct } from "constructs";
import { DataLakeStack } from "../stacks/DataLakeStack";
import { DefaultIdBuilder } from "../../utils/Naming";
import { EcsFargateServiceConstruct } from "../constructs/EcsFargateServiceConstruct";

import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import { Effect, PolicyStatement } from "aws-cdk-lib/aws-iam";

export interface ModelServingStageProps {
  dataLakeStack: DataLakeStack;
  modelRegistryTable: dynamodb.Table;
}

export class ModelServingStage extends Construct implements sfn.IChainable {

  readonly id: string;
  readonly startState: sfn.State;
  readonly endStates: sfn.INextable[];

  constructor(scope: Construct, id: string, props: ModelServingStageProps) {
    super(scope, id);

    this.id = id;

    const modelServingStageName = 'model-serving';
    const modelServingServiceId = DefaultIdBuilder.build(`${modelServingStageName}-service`);
    
    // Create persistent ECS Fargate service for model serving
    const modelServingService = new EcsFargateServiceConstruct(this, modelServingServiceId, {
      name: modelServingStageName,
      // IMPORTANT: the image path is relative to cdk.out
      imagePath: `../pipeline/${modelServingStageName}`,
      cpu: 2048, // Higher CPU for model inference
      memoryLimitMiB: 4096, // Higher memory for model loading
      portMappings: [{ containerPort: 8080 }], // HTTP port for inference API
      persistent: true, // Run as persistent service
      environment: {
        GOLD_BUCKET: props.dataLakeStack.goldBucket.bucketName,
        ARTIFACTS_BUCKET: props.dataLakeStack.artifactsBucket.bucketName,
        MODEL_REGISTRY_TABLE: props.modelRegistryTable.tableName,
        EXECUTION_MODE: 'standalone', // Running as persistent service
        PIPELINE_EXECUTION_ID: 'persistent-service',
        EXECUTION_START_TIME: 'continuous'
      }
    });

    // Grant permissions to read from gold bucket (for processed data)
    props.dataLakeStack.goldBucket.grantRead(modelServingService.service.taskDefinition.taskRole);
    
    // Grant permissions to read from artifacts bucket (for trained models)
    props.dataLakeStack.artifactsBucket.grantRead(modelServingService.service.taskDefinition.taskRole);
    
    // Grant permissions to read from model registry table
    const modelRegistryReadStatement = new PolicyStatement({
      effect: Effect.ALLOW,
      actions: [
        'dynamodb:GetItem',
        'dynamodb:Query',
        'dynamodb:Scan',
        'dynamodb:DescribeTable'
      ],
      resources: [props.modelRegistryTable.tableArn]
    });
    modelServingService.service.taskDefinition.taskRole.addToPrincipalPolicy(modelRegistryReadStatement);

    // Grant permissions to write inference results to gold bucket
    props.dataLakeStack.goldBucket.grantWrite(modelServingService.service.taskDefinition.taskRole);

    // Create ECS task for model serving initialization
    const modelServingTaskId = DefaultIdBuilder.build(`${modelServingStageName}-task`);
    const modelServingTask = new tasks.EcsRunTask(this, modelServingTaskId, {
      stateName: "Deploying model serving service",
      comment: "Initialize model serving service and load models - using REQUEST_RESPONSE since service runs persistently",
      cluster: modelServingService.service.cluster,
      taskDefinition: modelServingService.service.taskDefinition,
      integrationPattern: sfn.IntegrationPattern.REQUEST_RESPONSE,
      
      // Configure task parameters
      launchTarget: new tasks.EcsFargateLaunchTarget({
        platformVersion: ecs.FargatePlatformVersion.LATEST,
      }),
      
      // Add container overrides with execution context
      containerOverrides: [{
        containerDefinition: modelServingService.service.taskDefinition.defaultContainer!,
        environment: [
          { name: 'EXECUTION_MODE', value: 'step-functions' },
          { name: 'PIPELINE_EXECUTION_ID', value: sfn.JsonPath.stringAt('$$.Execution.Id') },
          { name: 'EXECUTION_START_TIME', value: sfn.JsonPath.stringAt('$$.Execution.StartTime') }
        ]
      }],
      
      // Result handling
      resultPath: '$.modelServingResult'
    });

    this.startState = modelServingTask;
    this.endStates = [modelServingTask];
  }
}
