import { APIGatewayProxyEvent, APIGatewayProxyResult, Context } from 'aws-lambda';
import { handleLambdaError, loadEnvVars } from '@wayweaver/lambda-utils';
import { ECSClient, RunTaskCommand } from '@aws-sdk/client-ecs';

interface StartRayTrainingRequest {
  executionId: string;
  executionStartTime?: string;
}

interface StepFunctionsPayload {
  executionId: string;
  executionStartTime?: string;
}

interface StartRayTrainingResponse {
  taskArn: string;
  status: string;
  message: string;
}

// Environment variables
const {
  ECS_CLUSTER_ARN,
  TASK_DEFINITION_ARN,
  CONTAINER_NAME,
  EKS_CLUSTER_NAME,
  RAY_NAMESPACE,
  GOLD_BUCKET,
  ARTIFACTS_BUCKET,
  MODEL_REGISTRY_TABLE,
  SUBNET_IDS,
  SECURITY_GROUP_IDS,
  ASSIGN_PUBLIC_IP
} = loadEnvVars([
  'ECS_CLUSTER_ARN',
  'TASK_DEFINITION_ARN',
  'CONTAINER_NAME',
  'EKS_CLUSTER_NAME',
  'RAY_NAMESPACE',
  'GOLD_BUCKET',
  'ARTIFACTS_BUCKET',
  'MODEL_REGISTRY_TABLE',
  'SUBNET_IDS',
  'SECURITY_GROUP_IDS',
  'ASSIGN_PUBLIC_IP'
]);

export const handler = async (
  event: APIGatewayProxyEvent | StartRayTrainingRequest | StepFunctionsPayload,
  context: Context
): Promise<APIGatewayProxyResult | StartRayTrainingResponse> => {
  console.log('Starting Ray training job handler');
  console.log('Event:', JSON.stringify(event, null, 2));

  try {
    // Validate required network configuration
    if (!SUBNET_IDS || !SECURITY_GROUP_IDS || !ASSIGN_PUBLIC_IP) {
      throw new Error('Missing required network configuration: SUBNET_IDS, SECURITY_GROUP_IDS, or ASSIGN_PUBLIC_IP');
    }

    // Extract data from event
    const requestData = 'body' in event ? JSON.parse(event.body || '{}') : event;
    const { executionId, executionStartTime } = requestData;
    
    if (!executionId) {
      throw new Error('executionId is required');
    }

    console.log(`Starting Ray training job for execution: ${executionId}`);
    console.log(`Network config - Subnets: ${SUBNET_IDS}, Security Groups: ${SECURITY_GROUP_IDS}, Public IP: ${ASSIGN_PUBLIC_IP}`);

    const ecsClient = new ECSClient({ region: process.env.AWS_REGION });

    // Create ECS task parameters
    const taskParams = {
      cluster: ECS_CLUSTER_ARN,
      taskDefinition: TASK_DEFINITION_ARN,
      launchType: 'FARGATE' as const,
      networkConfiguration: {
        awsvpcConfiguration: {
          subnets: SUBNET_IDS!.split(','),
          securityGroups: SECURITY_GROUP_IDS!.split(','),
          assignPublicIp: (ASSIGN_PUBLIC_IP === 'ENABLED' ? 'ENABLED' : 'DISABLED') as 'ENABLED' | 'DISABLED',
        },
      },
      overrides: {
        containerOverrides: [
          {
            name: CONTAINER_NAME,
            environment: [
              { name: 'EXECUTION_ID', value: executionId },
              { name: 'EXECUTION_START_TIME', value: executionStartTime || new Date().toISOString() },
              { name: 'EKS_CLUSTER_NAME', value: EKS_CLUSTER_NAME },
              { name: 'RAY_NAMESPACE', value: RAY_NAMESPACE },
              { name: 'GOLD_BUCKET', value: GOLD_BUCKET },
              { name: 'ARTIFACTS_BUCKET', value: ARTIFACTS_BUCKET },
              { name: 'MODEL_REGISTRY_TABLE', value: MODEL_REGISTRY_TABLE },
            ],
          },
        ],
      },
      tags: [
        { key: 'ExecutionId', value: executionId },
        { key: 'Project', value: 'MacroCausal' },
        { key: 'Stage', value: 'ModelTraining' },
        { key: 'StartTime', value: new Date().toISOString() },
      ],
    };

    // Start the ECS task
    const runTaskCommand = new RunTaskCommand(taskParams);
    const response = await ecsClient.send(runTaskCommand);

    if (!response.tasks || response.tasks.length === 0) {
      throw new Error('Failed to start ECS task for Ray training');
    }

    const taskArn = response.tasks[0].taskArn;
    if (!taskArn) {
      throw new Error('Failed to get task ARN from ECS response');
    }

    console.log(`Ray training task started successfully with ARN: ${taskArn}`);

    const responseData: StartRayTrainingResponse = {
      taskArn,
      status: 'STARTED',
      message: 'Ray training job started successfully',
    };

    return responseData;

  } catch (error) {
    handleLambdaError("StartRayTraining", error);
    throw error;
  }
};
