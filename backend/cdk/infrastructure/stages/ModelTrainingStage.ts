import { Construct } from "constructs";
import * as path from 'path';
import { DataLakeStack } from "../stacks/DataLakeStack";
import { DefaultIdBuilder } from "../../utils/Naming";
import { EksRayClusterConstruct } from "../constructs/EksRayClusterConstruct";
import { Code as LambdaCode, Function as LambdaFunction, ILayerVersion } from "aws-cdk-lib/aws-lambda"
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ecrAssets from 'aws-cdk-lib/aws-ecr-assets';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import { Duration, Stack } from 'aws-cdk-lib';
import { DEFAULT_LAMBDA_NODEJS_RUNTIME } from "@wayweaver/ariadne";
import { LambdaConfig } from "../configs/LambdaConfig";
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';

export interface ModelTrainingStageProps {
  dataLakeStack: DataLakeStack;
  commonUtilsLambdaLayer: ILayerVersion;
  ecsLambdaLayer: ILayerVersion;
  modelRegistryTable: dynamodb.Table;
}

export class ModelTrainingStage extends Construct implements sfn.IChainable {
  readonly id: string;
  readonly startState: sfn.State;
  readonly endStates: sfn.INextable[];

  constructor(scope: Construct, id: string, props: ModelTrainingStageProps) {
    super(scope, id);

    this.id = id;

    const modelTrainingStageName = 'model-training';

    // Create Docker image asset for model training code
    const modelTrainingImageId = DefaultIdBuilder.build('model-training-image');
    const modelTrainingImage = new ecrAssets.DockerImageAsset(this, modelTrainingImageId, {
      directory: `../pipeline/${modelTrainingStageName}`,
      platform: ecrAssets.Platform.LINUX_AMD64,
      buildArgs: {
        'PYTHON_VERSION': '3.10',
        'RAY_VERSION': '2.8.0',
        'TORCH_VERSION': '2.1.0'
      }
    });

    // Create EKS cluster for Ray training
    const rayClusterId = DefaultIdBuilder.build('ray-cluster');
    const rayCluster = new EksRayClusterConstruct(this, rayClusterId, {
      name: modelTrainingStageName,
      goldBucket: props.dataLakeStack.goldBucket,
      artifactsBucket: props.dataLakeStack.artifactsBucket,
      modelRegistryTable: props.modelRegistryTable,
    });

    // Create ECS cluster for Ray job orchestration
    const ecsClusterId = DefaultIdBuilder.build('ray-ecs-cluster');
    const ecsCluster = new ecs.Cluster(this, ecsClusterId, {
      enableFargateCapacityProviders: true
    });

    // Create ECS task definition for Ray training
    const taskDefinitionId = DefaultIdBuilder.build('ray-training-task');
    const taskDefinition = new ecs.FargateTaskDefinition(this, taskDefinitionId, {
      cpu: 2048,
      memoryLimitMiB: 4096,
      taskRole: this.createTaskRole(props),
      executionRole: this.createExecutionRole(),
    });

    // Add container to task definition
    const containerId = DefaultIdBuilder.build('ray-training-container');
    const container = taskDefinition.addContainer(containerId, {
      image: ecs.ContainerImage.fromDockerImageAsset(modelTrainingImage),
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: modelTrainingStageName }),
      environment: {
        EKS_CLUSTER_NAME: rayCluster.cluster.clusterName,
        RAY_NAMESPACE: rayCluster.rayNamespace,
        GOLD_BUCKET: props.dataLakeStack.goldBucket.bucketName,
        ARTIFACTS_BUCKET: props.dataLakeStack.artifactsBucket.bucketName,
        MODEL_REGISTRY_TABLE: props.modelRegistryTable.tableName,
      },
    });

    // Create Lambda function to start Ray training job
    const startTrainingLambdaId = DefaultIdBuilder.build('start-ray-training-lambda');
    const startTrainingLambda = new LambdaFunction(this, startTrainingLambdaId, {
      code: LambdaCode.fromAsset(path.join(__dirname, LambdaConfig.LAMBDA_CODE_RELATIVE_PATH)),
      handler: 'StartRayTraining.handler',
      runtime: DEFAULT_LAMBDA_NODEJS_RUNTIME,
      timeout: Duration.minutes(5),
      environment: {
        ECS_CLUSTER_ARN: ecsCluster.clusterArn,
        TASK_DEFINITION_ARN: taskDefinition.taskDefinitionArn,
        CONTAINER_NAME: containerId, // Pass the actual container name
        EKS_CLUSTER_NAME: rayCluster.cluster.clusterName,
        RAY_NAMESPACE: rayCluster.rayNamespace,
        GOLD_BUCKET: props.dataLakeStack.goldBucket.bucketName,
        ARTIFACTS_BUCKET: props.dataLakeStack.artifactsBucket.bucketName,
        MODEL_REGISTRY_TABLE: props.modelRegistryTable.tableName,
      },
      layers: [
        props.commonUtilsLambdaLayer,
        props.ecsLambdaLayer,
      ],
    });

    // Create Lambda function to check Ray training job status
    const checkTrainingStatusLambdaId = DefaultIdBuilder.build('check-ray-training-status-lambda');
    const checkTrainingStatusLambda = new LambdaFunction(this, checkTrainingStatusLambdaId, {
      code: LambdaCode.fromAsset(path.join(__dirname, LambdaConfig.LAMBDA_CODE_RELATIVE_PATH)),
      handler: 'CheckRayTrainingStatus.handler',
      runtime: DEFAULT_LAMBDA_NODEJS_RUNTIME,
      timeout: Duration.minutes(1),
      environment: {
        ECS_CLUSTER_ARN: ecsCluster.clusterArn
      },  
      layers: [
        props.commonUtilsLambdaLayer,
        props.ecsLambdaLayer,
      ],
    });

    // Grant ECS permissions to Lambda functions
    [startTrainingLambda, checkTrainingStatusLambda].forEach(lambdaFunc => {
      lambdaFunc.addToRolePolicy(new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'ecs:RunTask',
          'ecs:StopTask',
          'ecs:DescribeTasks',
          'ecs:ListTasks',
          'ecs:TagResource',
          'ecs:UntagResource',
          'ecs:ListTagsForResource',
        ],
        resources: ['*'],
      }));

      const passRoleArns = [taskDefinition.taskRole.roleArn, taskDefinition.executionRole?.roleArn].filter(Boolean) as string[];
      if (passRoleArns.length) {
        lambdaFunc.addToRolePolicy(new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: [
            'iam:PassRole',
          ],
          resources: passRoleArns,
        }));
      }
    });

    // Grant EKS permissions to Lambda functions
    [startTrainingLambda, checkTrainingStatusLambda].forEach(lambdaFunc => {
      lambdaFunc.addToRolePolicy(new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'eks:DescribeCluster',
        ],
        resources: [rayCluster.cluster.clusterArn],
      }));
    });

    // Create Step Functions tasks
    const startTrainingTaskId = DefaultIdBuilder.build(`${modelTrainingStageName}-start-task`);
    const startTrainingTask = new tasks.LambdaInvoke(this, startTrainingTaskId, {
      stateName: "Start Ray EKS training job",
      comment: 'Start Ray training job on EKS',
      lambdaFunction: startTrainingLambda,
      resultPath: '$.trainingStartResult',
      // Pass the execution ID from the Step Functions execution context
      payload: sfn.TaskInput.fromObject({
        executionId: sfn.JsonPath.stringAt('$$.Execution.Id'),
        executionStartTime: sfn.JsonPath.stringAt('$$.Execution.StartTime')
      })
    });

    const checkStatusTaskId = DefaultIdBuilder.build(`${modelTrainingStageName}-check-status-task`);
    const checkStatusTask = new tasks.LambdaInvoke(this, checkStatusTaskId, {
      stateName: "Polling Ray EKS job status",
      comment: 'Check Ray training job status',
      lambdaFunction: checkTrainingStatusLambda,
      resultPath: '$.trainingStatusResult',
      // Pass the task ARN from the previous stage result
      payload: sfn.TaskInput.fromObject({
        taskArn: sfn.JsonPath.stringAt('$.trainingStartResult.Payload.taskArn'),
        executionId: sfn.JsonPath.stringAt('$$.Execution.Id')
      })
    });

    // Create wait state
    const waitStateId = DefaultIdBuilder.build(`${modelTrainingStageName}-wait`);
    const waitState = new sfn.Wait(this, waitStateId, {
      stateName: "Waiting for Ray EKS job to complete",
      time: sfn.WaitTime.duration(Duration.seconds(30)),
    });

    // Connect wait state back to check status task
    waitState.next(checkStatusTask);

    const successStateId = DefaultIdBuilder.build(`${modelTrainingStageName}-success`);
    const successState = new sfn.Pass(this, successStateId, {
      stateName: "Model training stage succeeded",
      comment: 'Model training stage finished successfully',
    });

    const failureStateId = DefaultIdBuilder.build(`${modelTrainingStageName}-failed`);
    const failureState = new sfn.Fail(this, failureStateId, {
      stateName: "Model training stage failed",
      comment: 'Model training stage encountered an error',
    });

    // Create a choice state to check job status
    const jobStatusChoiceId = DefaultIdBuilder.build(`${modelTrainingStageName}-job-complete-choice`);
    const jobStatusChoice = new sfn.Choice(this, jobStatusChoiceId, {
      stateName: "Ray EKS job status?",
    })
      .when(sfn.Condition.stringEquals('$.trainingStatusResult.Payload.status', 'SUCCESS'), successState)
      .when(sfn.Condition.stringEquals('$.trainingStatusResult.Payload.status', 'FAILED'), failureState)
      .otherwise(waitState);

    // Build the training workflow
    startTrainingTask.next(checkStatusTask).next(jobStatusChoice);

    this.startState = startTrainingTask;
    this.endStates = [successState];
  }

  private createTaskRole(props: ModelTrainingStageProps): iam.Role {
    const taskRoleId = DefaultIdBuilder.build('ray-training-task-role');
    const taskRole = new iam.Role(this, taskRoleId, {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
    });

    // Grant S3 access
    props.dataLakeStack.goldBucket.grantRead(taskRole);
    props.dataLakeStack.artifactsBucket.grantReadWrite(taskRole);

    // Grant DynamoDB access for model registry
    taskRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'dynamodb:GetItem',
        'dynamodb:PutItem',
        'dynamodb:UpdateItem',
        'dynamodb:DeleteItem',
        'dynamodb:Query',
        'dynamodb:Scan',
      ],
      resources: [props.modelRegistryTable.tableArn],
    }));

    // Grant EKS access for Ray cluster management
    taskRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'eks:DescribeCluster',
      ],
      resources: ['*'],
    }));

    // Grant ECS permissions for task operations
    taskRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'ecs:TagResource',
        'ecs:UntagResource',
        'ecs:ListTagsForResource',
        'ecs:DescribeTasks',
        'ecs:DescribeTaskDefinition',
      ],
      resources: ['*'],
    }));

    return taskRole;
  }

  private createExecutionRole(): iam.Role {
    const executionRoleId = DefaultIdBuilder.build('ray-training-execution-role');
    const executionRole = new iam.Role(this, executionRoleId, {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonECSTaskExecutionRolePolicy'),
      ],
    });

    // Add ECS permissions for tagging resources and other operations
    executionRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'ecs:TagResource',
        'ecs:UntagResource',
        'ecs:ListTagsForResource',
        'ecs:DescribeTasks',
        'ecs:DescribeTaskDefinition',
        'ecs:DescribeServices',
        'ecs:DescribeClusters',
        'ecs:UpdateService',
        'ecs:UpdateTaskSet',
        'ecs:UpdateTaskDefinition',
        'ecs:CreateService',
        'ecs:DeleteService',
        'ecs:CreateTaskSet',
        'ecs:DeleteTaskSet',
      ],
      resources: ['*'],
    }));

    // Add CloudWatch Logs permissions for ECS task logging
    executionRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'logs:CreateLogGroup',
        'logs:CreateLogStream',
        'logs:PutLogEvents',
        'logs:DescribeLogGroups',
        'logs:DescribeLogStreams',
      ],
      resources: ['*'],
    }));

    return executionRole;
  }
}
