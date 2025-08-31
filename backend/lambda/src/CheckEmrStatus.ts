import { APIGatewayProxyEvent, APIGatewayProxyResult, Context } from 'aws-lambda';
import { EMRServerlessManager } from './EmrUtils';
import { handleLambdaError, loadEnvVars } from '@wayweaver/lambda-utils';

interface CheckStatusRequest {
  jobRunId: string;
}

interface CheckStatusResponse {
  jobRunId: string;
  status: string;
  message: string;
  details?: {
    state: string;
    stateDetails?: string;
    createdAt?: string;
    updatedAt?: string;
    totalResourceUtilization?: {
      vCPUHour?: number;
      memoryGBHour?: number;
      storageGBHour?: number;
    };
  };
}

// Environment variables
const {
  EMR_APPLICATION_ID
} = loadEnvVars([
  'EMR_APPLICATION_ID'
]);

export const handler = async (
  event: APIGatewayProxyEvent | CheckStatusRequest,
  context: Context
): Promise<APIGatewayProxyResult | CheckStatusResponse> => {
  console.log('Checking EMR Serverless job status');
  console.log('Event:', JSON.stringify(event, null, 2));
  console.log('Context:', JSON.stringify(context, null, 2));

  try {
    // Extract data from event
    const requestData = 'body' in event ? JSON.parse(event.body || '{}') : event;
    const { jobRunId } = requestData;
    
    if (!jobRunId) {
      throw new Error('jobRunId is required');
    }

    console.log(`Checking status for EMR job: ${jobRunId}`);

    // Create EMR manager and get job status
    const emrManager = new EMRServerlessManager(EMR_APPLICATION_ID);
    const jobStatus = await emrManager.getJobStatus(jobRunId);
    
    // Get simplified status for Step Functions
    const simplifiedStatus = emrManager.getSimplifiedStatus(jobStatus);
    
    console.log(`Job status: ${jobStatus.state}`);
    if (jobStatus.stateDetails) {
      console.log(`State details: ${jobStatus.stateDetails}`);
    }

    const response: CheckStatusResponse = {
      jobRunId,
      status: simplifiedStatus,
      message: `Job status: ${jobStatus.state}`,
      details: {
        state: jobStatus.state,
        stateDetails: jobStatus.stateDetails,
        createdAt: jobStatus.createdAt?.toISOString(),
        updatedAt: jobStatus.updatedAt?.toISOString(),
        totalResourceUtilization: jobStatus.totalResourceUtilization ? {
          vCPUHour: jobStatus.totalResourceUtilization.vCPUHour,
          memoryGBHour: jobStatus.totalResourceUtilization.memoryGBHour,
          storageGBHour: jobStatus.totalResourceUtilization.storageGBHour
        } : undefined
      }
    };

    // Log resource utilization if available
    if (jobStatus.totalResourceUtilization) {
      console.log('Resource utilization:', {
        vCPUHour: jobStatus.totalResourceUtilization.vCPUHour,
        memoryGBHour: jobStatus.totalResourceUtilization.memoryGBHour,
        storageGBHour: jobStatus.totalResourceUtilization.storageGBHour
      });
    }

    // Return response to Step Functions
    return response;

  } catch (error) {
    handleLambdaError("CheckEmrStatus", error);
    // Rethrow error to Step Functions
    throw error;
  }
};
