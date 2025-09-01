import { EMRServerlessClient, StartJobRunCommand, GetJobRunCommand, CancelJobRunCommand } from '@aws-sdk/client-emr-serverless';

const emrClient = new EMRServerlessClient({ region: process.env.AWS_REGION });

export interface EMRJobConfig {
  applicationId: string;
  executionRoleArn: string;
  entryPoint: string;
  entryPointArguments: string[];
  sparkSubmitParameters: string[];
  jobName: string;
  tags?: Record<string, string>;
}

export interface EMRJobStatus {
  jobRunId: string;
  state: string;
  stateDetails?: string;
  createdAt?: Date;
  updatedAt?: Date;
  totalResourceUtilization?: {
    vCPUHour?: number;
    memoryGBHour?: number;
    storageGBHour?: number;
  };
}

export class EMRServerlessManager {
  private applicationId: string;

  constructor(applicationId: string) {
    this.applicationId = applicationId;
  }

  /**
   * Start an EMR Serverless job
   */
  async startJob(config: EMRJobConfig): Promise<string> {
    try {
      console.log(`Starting EMR job: ${config.jobName}`);

      const startJobCommand = new StartJobRunCommand({
        applicationId: config.applicationId,
        executionRoleArn: config.executionRoleArn,
        jobDriver: {
          sparkSubmit: {
            entryPoint: config.entryPoint,
            entryPointArguments: config.entryPointArguments,
            sparkSubmitParameters: config.sparkSubmitParameters.join(' ')
          }
        },
        configurationOverrides: {
          applicationConfiguration: [
            {
              classification: 'spark-defaults',
              properties: {
                'spark.driver.memory': '16g',
                'spark.driver.maxResultSize': '4g',
                'spark.sql.adaptive.enabled': 'true',
                'spark.sql.adaptive.coalescePartitions.enabled': 'true',
                'spark.sql.adaptive.skewJoin.enabled': 'true',
                'spark.sql.parquet.compression': 'snappy'
              }
            }
          ]
        },
        name: config.jobName,
        tags: config.tags
      });

      const response = await emrClient.send(startJobCommand);
      
      if (!response.jobRunId) {
        throw new Error('Failed to get job run ID from EMR Serverless response');
      }

      console.log(`EMR job started successfully with ID: ${response.jobRunId}`);
      return response.jobRunId;

    } catch (error) {
      console.error('Error starting EMR job:', error);
      throw error;
    }
  }

  /**
   * Get the status of an EMR Serverless job
   */
  async getJobStatus(jobRunId: string): Promise<EMRJobStatus> {
    try {
      console.log(`Getting status for EMR job: ${jobRunId}`);

      const getJobCommand = new GetJobRunCommand({
        applicationId: this.applicationId,
        jobRunId
      });

      const response = await emrClient.send(getJobCommand);
      const jobRun = response.jobRun;

      if (!jobRun) {
        throw new Error(`Job run not found: ${jobRunId}`);
      }

      const jobStatus: EMRJobStatus = {
        jobRunId,
        state: jobRun.state || 'UNKNOWN',
        stateDetails: jobRun.stateDetails,
        createdAt: jobRun.createdAt,
        updatedAt: jobRun.updatedAt,
        totalResourceUtilization: jobRun.totalResourceUtilization
      };

      console.log(`Job status: ${jobStatus.state}`);
      return jobStatus;

    } catch (error) {
      console.error('Error getting EMR job status:', error);
      throw error;
    }
  }

  /**
   * Cancel an EMR Serverless job
   */
  async cancelJob(jobRunId: string): Promise<void> {
    try {
      console.log(`Cancelling EMR job: ${jobRunId}`);

      const cancelJobCommand = new CancelJobRunCommand({
        applicationId: this.applicationId,
        jobRunId
      });

      await emrClient.send(cancelJobCommand);
      console.log(`EMR job cancelled successfully: ${jobRunId}`);

    } catch (error) {
      console.error('Error cancelling EMR job:', error);
      throw error;
    }
  }

  /**
   * Wait for job completion with timeout
   */
  async waitForJobCompletion(jobRunId: string, timeoutMinutes: number = 60): Promise<EMRJobStatus> {
    const startTime = Date.now();
    const timeoutMs = timeoutMinutes * 60 * 1000;
    const pollIntervalMs = 30000; // 30 seconds

    while (true) {
      const jobStatus = await this.getJobStatus(jobRunId);
      
      // Check if job is complete
      if (jobStatus.state === 'SUCCESS' || jobStatus.state === 'FAILED' || jobStatus.state === 'CANCELLED') {
        return jobStatus;
      }

      // Check timeout
      if (Date.now() - startTime > timeoutMs) {
        throw new Error(`Job timeout after ${timeoutMinutes} minutes. Current status: ${jobStatus.state}`);
      }

      // Wait before next poll
      await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
    }
  }

  /**
   * Get simplified status for Step Functions
   */
  getSimplifiedStatus(jobStatus: EMRJobStatus): string {
    switch (jobStatus.state) {
      case 'SUBMITTED':
      case 'PENDING':
      case 'SCHEDULED':
      case 'RUNNING':
        return 'RUNNING';
      case 'SUCCESS':
        return 'SUCCESS';
      case 'FAILED':
      case 'CANCELLED':
        return 'FAILED';
      default:
        return 'UNKNOWN';
    }
  }
}

/**
 * Create default EMR job configuration for data processing
 */
export function createDataProcessingJobConfig(
  executionId: string,
  applicationId: string,
  executionRoleArn: string,
  bronzeBucket: string,
  silverBucket: string,
  goldBucket: string
): EMRJobConfig {
  return {
    applicationId,
    executionRoleArn,
    entryPoint: '/opt/spark/work-dir/main.py',
    entryPointArguments: [
      '--bronze-bucket', bronzeBucket,
      '--silver-bucket', silverBucket,
      '--gold-bucket', goldBucket,
      '--execution-id', executionId
    ],
    sparkSubmitParameters: [
      '--conf', 'spark.sql.adaptive.enabled=true',
      '--conf', 'spark.sql.adaptive.coalescePartitions.enabled=true',
      '--conf', 'spark.sql.adaptive.skewJoin.enabled=true',
      '--conf', 'spark.sql.parquet.compression=snappy',
      '--conf', 'spark.executor.cores=4',
      '--conf', 'spark.executor.memory=16g',
      '--conf', 'spark.driver.cores=4',
      '--conf', 'spark.driver.memory=16g'
    ],
    jobName: `data-processing-${executionId}`,
    tags: {
      'ExecutionId': executionId,
      'Project': 'MacroCausal',
      'Stage': 'DataProcessing',
      'StartTime': new Date().toISOString()
    }
  };
}
