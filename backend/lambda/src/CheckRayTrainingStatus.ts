import { APIGatewayProxyEvent, APIGatewayProxyResult, Context } from 'aws-lambda';
import { handleLambdaError, loadEnvVars } from '@wayweaver/lambda-utils';
import { ECSClient, DescribeTasksCommand } from '@aws-sdk/client-ecs';

interface CheckRayTrainingStatusRequest {
  taskArn: string;
}

interface CheckRayTrainingStatusResponse {
  status: string;
  message: string;
  details?: any;
}

// Environment variables
const {
  ECS_CLUSTER_ARN,
  AWS_REGION
} = loadEnvVars([
  'ECS_CLUSTER_ARN',
  'AWS_REGION'
]);

export const handler = async (
  event: APIGatewayProxyEvent | CheckRayTrainingStatusRequest,
  context: Context
): Promise<APIGatewayProxyResult | CheckRayTrainingStatusResponse> => {
  console.log('Checking Ray training job status handler');
  console.log('Event:', JSON.stringify(event, null, 2));

  try {
    // Extract data from event
    const requestData = 'body' in event ? JSON.parse(event.body || '{}') : event;
    const { taskArn } = requestData;
    
    if (!taskArn) {
      throw new Error('taskArn is required');
    }

    console.log(`Checking status for Ray training task: ${taskArn}`);

    const ecsClient = new ECSClient({ region: AWS_REGION });

    // Check ECS task status
    const describeTasksCommand = new DescribeTasksCommand({
      cluster: ECS_CLUSTER_ARN,
      tasks: [taskArn],
    });

    const ecsResponse = await ecsClient.send(describeTasksCommand);
    
    if (!ecsResponse.tasks || ecsResponse.tasks.length === 0) {
      throw new Error(`Task not found: ${taskArn}`);
    }

    const task = ecsResponse.tasks[0];
    const taskStatus = task.lastStatus || 'UNKNOWN';
    const desiredStatus = task.desiredStatus || 'UNKNOWN';

    console.log(`ECS task status: ${taskStatus}, desired status: ${desiredStatus}`);

    // Determine overall status
    let status: string;
    let message: string;
    let details: any = { ecsStatus: taskStatus, desiredStatus };

    if (taskStatus === 'STOPPED') {
      // Check exit code to determine success/failure
      const container = task.containers?.[0];
      if (container?.exitCode === 0) {
        status = 'SUCCESS';
        message = 'Ray training job completed successfully';
      } else {
        status = 'FAILED';
        message = `Ray training job failed with exit code: ${container?.exitCode}`;
        details.exitCode = container?.exitCode;
        details.reason = container?.reason;
      }
    } else if (taskStatus === 'RUNNING') {
      status = 'RUNNING';
      message = 'Ray training job is running';
    } else if (taskStatus === 'PENDING') {
      status = 'RUNNING';
      message = 'Ray training job is starting';
    } else {
      status = 'UNKNOWN';
      message = `Ray training job has unknown status: ${taskStatus}`;
    }

    const responseData: CheckRayTrainingStatusResponse = {
      status,
      message,
      details,
    };

    console.log(`Ray training status: ${status}`);
    return responseData;

  } catch (error) {
    handleLambdaError("CheckRayTrainingStatus", error);
    throw error;
  }
};
