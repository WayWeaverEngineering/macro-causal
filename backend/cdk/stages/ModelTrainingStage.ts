import { Construct } from "constructs";
import * as path from 'path';
import { DataLakeStack } from "../stacks/DataLakeStack";
import { AWS_LAMBDA_LAYERS, ConstructIdBuilder, PrebuiltLambdaLayers, UTILS_LAMBDA_LAYERS } from '@wayweaver/ariadne';
import { MACRO_CAUSAL_CONSTANTS } from "../utils/Constants";
import { EksRayClusterConstruct } from "../infrastructure/constructs/EksRayClusterConstruct";
import { Code as LambdaCode, Function as LambdaFunction } from "aws-cdk-lib/aws-lambda"
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ecrAssets from 'aws-cdk-lib/aws-ecr-assets';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import { Duration } from 'aws-cdk-lib';
import { DEFAULT_LAMBDA_NODEJS_RUNTIME } from "@wayweaver/ariadne";
import { LambdaConfig } from "../configs/LambdaConfig";
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as ec2 from 'aws-cdk-lib/aws-ec2';

export interface ModelTrainingStageProps {
  idBuilder: ConstructIdBuilder;
  dataLakeStack: DataLakeStack;
  lambdaLayersStack: PrebuiltLambdaLayers;
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
    const modelTrainingImageId = props.idBuilder.build('model-training-image');
    const modelTrainingImage = new ecrAssets.DockerImageAsset(this, modelTrainingImageId, {
      directory: `../pipeline/${modelTrainingStageName}`,
      platform: ecrAssets.Platform.LINUX_AMD64,
      buildArgs: {
        'PYTHON_VERSION': '3.10',
        'RAY_VERSION': MACRO_CAUSAL_CONSTANTS.RAY.VERSION,
        'TORCH_VERSION': '2.1.0'
      }
    });

    // Create EKS cluster for Ray training
    const rayClusterId = props.idBuilder.build('ray-cluster');
    const rayCluster = new EksRayClusterConstruct(this, rayClusterId, {
      name: modelTrainingStageName,
      goldBucket: props.dataLakeStack.goldBucket,
      artifactsBucket: props.dataLakeStack.artifactsBucket,
      modelRegistryTable: props.modelRegistryTable,
    });

    // Create ECS cluster for Ray job orchestration
    // IMPORTANT: ECS cluster must be in the SAME VPC as EKS cluster for Fargate tasks
    // to reach the private EKS API endpoint without NAT gateways
    const ecsClusterId = props.idBuilder.build('ray-ecs-cluster');
    const ecsCluster = new ecs.Cluster(this, ecsClusterId, {
      vpc: rayCluster.cluster.vpc,             // <— SAME VPC AS EKS
      enableFargateCapacityProviders: true,
    });

    // Add a Security Group for Fargate tasks
    // This SG allows outbound access to VPC endpoints (EKS, ECR, Logs, S3, STS, EC2)
    // No inbound rules needed - tasks only need to reach AWS services via endpoints
    const taskSg = new ec2.SecurityGroup(this, props.idBuilder.build('ray-ecs-task-sg'), {
      vpc: ecsCluster.vpc,
      description: 'SG for Ray training Fargate tasks',
      allowAllOutbound: true, // Needed for VPC endpoints, ECR, CW Logs, etc.
    });

    // Select the PUBLIC subnets
    // These subnets have internet access through the Internet Gateway
    // This ensures Fargate tasks can reach the public EKS API endpoint and pull images from ECR
    const publicSubnetIds = ecsCluster.vpc.selectSubnets({
      subnetType: ec2.SubnetType.PUBLIC,
    }).subnetIds;

    // Create ECS task definition for Ray training
    const taskDefinitionId = props.idBuilder.build('ray-training-task');
    const taskDefinition = new ecs.FargateTaskDefinition(this, taskDefinitionId, {
      cpu: 2048,
      memoryLimitMiB: 4096,
      taskRole: this.createTaskRole(props),
      executionRole: this.createExecutionRole(props.idBuilder),
    });

    // Add container to task definition
    const containerId = props.idBuilder.build('ray-training-container');
    const container = taskDefinition.addContainer(containerId, {
      image: ecs.ContainerImage.fromDockerImageAsset(modelTrainingImage),
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: modelTrainingStageName }),
      environment: {
        EKS_CLUSTER_NAME: rayCluster.cluster.clusterName,
        RAY_NAMESPACE: rayCluster.rayNamespace,
        GOLD_BUCKET: props.dataLakeStack.goldBucket.bucketName,
        ARTIFACTS_BUCKET: props.dataLakeStack.artifactsBucket.bucketName,
        MODEL_REGISTRY_TABLE: props.modelRegistryTable.tableName,
        SUBNET_IDS: publicSubnetIds.join(','),
        SECURITY_GROUP_IDS: taskSg.securityGroupId,
        ASSIGN_PUBLIC_IP: 'ENABLED', // public subnets → enable public IP for internet access
      },
    });

    // Create Lambda function to start Ray training job
    const startTrainingLambdaId = props.idBuilder.build('start-ray-training-lambda');
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
        SUBNET_IDS: publicSubnetIds.join(','),
        SECURITY_GROUP_IDS: taskSg.securityGroupId,
        ASSIGN_PUBLIC_IP: 'ENABLED', // public subnets → enable public IP for internet access
      },
      layers: [
        props.lambdaLayersStack.getLayer(AWS_LAMBDA_LAYERS.AWS_ECS_LAMBDA_LAYER),
        props.lambdaLayersStack.getLayer(UTILS_LAMBDA_LAYERS.LAMBDA_UTILS_LAMBDA_LAYER),
      ],
    });

    // Create Lambda function to check Ray training job status
    const checkTrainingStatusLambdaId = props.idBuilder.build('check-ray-training-status-lambda');
    const checkTrainingStatusLambda = new LambdaFunction(this, checkTrainingStatusLambdaId, {
      code: LambdaCode.fromAsset(path.join(__dirname, LambdaConfig.LAMBDA_CODE_RELATIVE_PATH)),
      handler: 'CheckRayTrainingStatus.handler',
      runtime: DEFAULT_LAMBDA_NODEJS_RUNTIME,
      timeout: Duration.minutes(1),
      environment: {
        ECS_CLUSTER_ARN: ecsCluster.clusterArn,
        SUBNET_IDS: publicSubnetIds.join(','),
        SECURITY_GROUP_IDS: taskSg.securityGroupId,
        ASSIGN_PUBLIC_IP: 'ENABLED', // public subnets → enable public IP for internet access
      },  
      layers: [
        props.lambdaLayersStack.getLayer(AWS_LAMBDA_LAYERS.AWS_ECS_LAMBDA_LAYER),
        props.lambdaLayersStack.getLayer(UTILS_LAMBDA_LAYERS.LAMBDA_UTILS_LAMBDA_LAYER),
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
    const startTrainingTaskId = props.idBuilder.build(`${modelTrainingStageName}-start-task`);
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

    const checkStatusTaskId = props.idBuilder.build(`${modelTrainingStageName}-check-status-task`);
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
    const waitStateId = props.idBuilder.build(`${modelTrainingStageName}-wait`);
    const waitState = new sfn.Wait(this, waitStateId, {
      stateName: "Waiting for Ray EKS job",
      time: sfn.WaitTime.duration(Duration.seconds(30)),
    });

    // Connect wait state back to check status task
    waitState.next(checkStatusTask);

    const successStateId = props.idBuilder.build(`${modelTrainingStageName}-success`);
    const successState = new sfn.Pass(this, successStateId, {
      stateName: "Model training stage succeeded",
      comment: 'Model training stage finished successfully',
    });

    const failureStateId = props.idBuilder.build(`${modelTrainingStageName}-failed`);
    const failureState = new sfn.Fail(this, failureStateId, {
      stateName: "Model training stage failed",
      comment: 'Model training stage encountered an error',
    });

    // Create a choice state to check job status
    const jobStatusChoiceId = props.idBuilder.build(`${modelTrainingStageName}-job-complete-choice`);
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
    const taskRoleId = props.idBuilder.build('ray-training-task-role');
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

  private createExecutionRole(idBuilder: ConstructIdBuilder): iam.Role {
    const executionRoleId = idBuilder.build('ray-training-execution-role');
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
