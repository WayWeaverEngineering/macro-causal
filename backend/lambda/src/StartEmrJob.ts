import { APIGatewayProxyEvent, APIGatewayProxyResult, Context } from 'aws-lambda';
import { EMRServerlessManager, createDataProcessingJobConfig } from './EmrUtils';
import { handleLambdaError, loadEnvVars } from '@wayweaver/lambda-utils';

interface StartJobRequest {
  executionId: string;
  executionStartTime: string;
}

interface StartJobResponse {
  jobRunId: string;
  status: string;
  message: string;
}

// Environment variables
const {
  EMR_APPLICATION_ID,
  EMR_EXECUTION_ROLE_ARN,
  BRONZE_BUCKET,
  SILVER_BUCKET,
  GOLD_BUCKET
} = loadEnvVars([
  'EMR_APPLICATION_ID',
  'EMR_EXECUTION_ROLE_ARN',
  'BRONZE_BUCKET',
  'SILVER_BUCKET',
  'GOLD_BUCKET'
]);

export const handler = async (
  event: APIGatewayProxyEvent | StartJobRequest,
  context: Context
): Promise<APIGatewayProxyResult | StartJobResponse> => {
  console.log('Starting EMR Serverless job handler');
  console.log('Event:', JSON.stringify(event, null, 2));
  console.log('Context:', JSON.stringify(context, null, 2));

  try {
    // Extract data from event
    const requestData = 'body' in event ? JSON.parse(event.body || '{}') : event;
    const { executionId, executionStartTime } = requestData;
    
    if (!executionId) {
      throw new Error('executionId is required');
    }

    console.log(`Starting EMR Serverless job for execution: ${executionId}`);

    // Create EMR manager and job configuration
    const emrManager = new EMRServerlessManager(EMR_APPLICATION_ID);
    const jobConfig = createDataProcessingJobConfig(
      executionId,
      EMR_APPLICATION_ID,
      EMR_EXECUTION_ROLE_ARN,
      BRONZE_BUCKET,
      SILVER_BUCKET,
      GOLD_BUCKET
    );
    
    // Add execution start time to tags if provided
    if (executionStartTime) {
      jobConfig.tags = { ...jobConfig.tags, 'ExecutionStartTime': executionStartTime };
    }

    // Start the job
    const jobRunId = await emrManager.startJob(jobConfig);

    const response: StartJobResponse = {
      jobRunId,
      status: 'STARTED',
      message: 'EMR job started successfully'
    };

    // Return response to Step Functions
    return response;

  } catch (error) {
    handleLambdaError("StartEmrJob", error);
    // Rethrow error to Step Functions
    throw error;
  }
};
