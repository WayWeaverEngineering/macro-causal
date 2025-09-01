import { SQSEvent, SQSRecord, Context } from "aws-lambda";
import { parseSqsRecord } from "@wayweaver/aws-queue";
import { loadEnvVars } from "@wayweaver/lambda-utils";
import { ExecutionStep } from "./models/AgentModels";
import { UpdateCommand, GetCommand } from '@aws-sdk/lib-dynamodb';
import { AnalysisExecution, AnalysisMessage } from "./models/AnalysisModels";
import { DbClient } from "./clients/DbClient";
import { initializeUuidPolyfill } from "./utils/UuidPolyfill";

// Initialize UUID polyfill to fix v6 function issue in older UUID versions
initializeUuidPolyfill();

// Helper function to remove undefined values from objects
function removeUndefinedValues(obj: any): any {
  if (obj === null || obj === undefined) {
    return obj;
  }
  
  if (Array.isArray(obj)) {
    return obj.map(removeUndefinedValues).filter(item => item !== undefined);
  }
  
  if (typeof obj === 'object') {
    const cleaned: any = {};
    for (const [key, value] of Object.entries(obj)) {
      if (value !== undefined) {
        cleaned[key] = removeUndefinedValues(value);
      }
    }
    return cleaned;
  }
  
  return obj;
}

// Environment variables
const {
  ANALYSIS_EXECUTIONS_TABLE,
} = loadEnvVars([
  'ANALYSIS_EXECUTIONS_TABLE',
]);

// Initialize AWS services
const dynamoDBDocumentClient = DbClient.getInstance();

// Helper function to update execution status
async function updateExecutionStatus(
  executionId: string, 
  updates: Partial<AnalysisExecution>
): Promise<void> {
  const updateExpression: string[] = [];
  const expressionAttributeValues: any = {};
  const expressionAttributeNames: any = {};

  // Build update expression dynamically
  Object.entries(updates).forEach(([key, value]) => {
    if (key === 'executionId') return; // Skip primary key
    if (value === undefined) return; // Skip undefined values
    
    const attributeName = `#${key}`;
    const attributeValue = `:${key}`;
    
    updateExpression.push(`${attributeName} = ${attributeValue}`);
    expressionAttributeNames[attributeName] = key;
    expressionAttributeValues[attributeValue] = value;
  });

  // Always update the updatedAt timestamp
  updateExpression.push('#updatedAt = :updatedAt');
  expressionAttributeNames['#updatedAt'] = 'updatedAt';
  expressionAttributeValues[':updatedAt'] = new Date().toISOString();

  await dynamoDBDocumentClient.send(new UpdateCommand({
    TableName: ANALYSIS_EXECUTIONS_TABLE,
    Key: { executionId },
    UpdateExpression: `SET ${updateExpression.join(', ')}`,
    ExpressionAttributeNames: expressionAttributeNames,
    ExpressionAttributeValues: expressionAttributeValues
  }));
}

// Helper function to update execution step
async function updateExecutionStep(
  executionId: string, 
  step: ExecutionStep
): Promise<void> {
  console.log(`Updating execution step for ${executionId}:`, step);
  
  // First, try to get the current steps to see if the field exists
  try {
    const result = await dynamoDBDocumentClient.send(new GetCommand({
      TableName: ANALYSIS_EXECUTIONS_TABLE,
      Key: { executionId },
      ProjectionExpression: 'steps'
    }));

    const currentSteps = (result.Item as any)?.steps || [];
    console.log(`Current steps for ${executionId}:`, currentSteps);
    
    const updatedSteps = [...currentSteps, step];
    console.log(`Updated steps for ${executionId}:`, updatedSteps);
    
    // Serialize the step object to handle Date objects properly for DynamoDB
    const serializedStep = {
      ...step,
      startTime: step.startTime ? step.startTime.toISOString() : undefined,
      endTime: step.endTime ? step.endTime.toISOString() : undefined
    };
    
    // Update with the new step
    await dynamoDBDocumentClient.send(new UpdateCommand({
      TableName: ANALYSIS_EXECUTIONS_TABLE,
      Key: { executionId },
      UpdateExpression: 'SET currentStep = :step, steps = :steps',
      ExpressionAttributeValues: {
        ':step': serializedStep,
        ':steps': updatedSteps.map(s => ({
          ...s,
          startTime: s.startTime ? (typeof s.startTime === 'string' ? s.startTime : s.startTime.toISOString()) : undefined,
          endTime: s.endTime ? (typeof s.endTime === 'string' ? s.endTime : s.endTime.toISOString()) : undefined
        }))
      }
    }));
    
    console.log(`Successfully updated execution step for ${executionId}`);
  } catch (error) {
    console.error(`Error updating execution step for ${executionId}:`, error);
    throw error;
  }
}

// Placeholder function for macro-causal analysis processing
async function processMacroCausalAnalysis(query: string): Promise<any> {
  // This is a placeholder implementation
  // In the real implementation, this would:
  // 1. Parse the query to understand what type of analysis is needed
  // 2. Collect relevant macroeconomic data
  // 3. Apply causal inference models (e.g., X-Learner, Double ML)
  // 4. Generate insights and policy implications
  
  console.log(`Processing macro-causal analysis for query: ${query}`);
  
  // Simulate processing time
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // Return placeholder result
  return {
    success: true,
    message: "Macro-causal analysis completed",
    analysis: {
      insight_type: 'treatment_effect',
      summary: "Placeholder analysis result - implement actual macro-causal logic here",
      magnitude: 0.15,
      direction: 'positive',
      confidence: 0.85,
      policy_implications: ["Implement monitoring of key indicators", "Consider policy adjustments"],
      limitations: ["Limited historical data", "Assumes ceteris paribus"],
      data_sources: ["Federal Reserve Economic Data", "Bureau of Labor Statistics"],
      methodology: "X-Learner with Double ML approach"
    },
    metadata: {
      executionTime: 2000,
      analysisType: "treatment_effect",
      complexity: "moderate"
    }
  };
}

// Process a single analysis record
async function processAnalysisRecord(record: SQSRecord): Promise<void> {
  const message = parseSqsRecord(record) as AnalysisMessage;
  const { executionId, query } = message;

  console.log(`Processing macro-causal analysis: ${executionId}`);

  try {
    // Update status to running
    await updateExecutionStatus(executionId, {
      status: 'running'
    });

    // Create initial step
    const initialStep: ExecutionStep = {
      stepId: 'step_1',
      stepName: 'Query Analysis',
      description: 'Analyzing user query and determining analysis requirements',
      status: 'in_progress',
      startTime: new Date(),
      metadata: { query }
    };
    
    await updateExecutionStep(executionId, initialStep);
    
    // Simulate query analysis completion
    await new Promise(resolve => setTimeout(resolve, 1000));
    initialStep.status = 'completed';
    initialStep.endTime = new Date();
    await updateExecutionStep(executionId, initialStep);

    // Create data collection step
    const dataStep: ExecutionStep = {
      stepId: 'step_2',
      stepName: 'Data Collection',
      description: 'Collecting relevant macroeconomic data and policy information',
      status: 'in_progress',
      startTime: new Date(),
      metadata: { dataSources: ['FRED', 'BLS', 'CBO'] }
    };
    
    await updateExecutionStep(executionId, dataStep);
    
    // Simulate data collection
    await new Promise(resolve => setTimeout(resolve, 1500));
    dataStep.status = 'completed';
    dataStep.endTime = new Date();
    await updateExecutionStep(executionId, dataStep);

    // Create model execution step
    const modelStep: ExecutionStep = {
      stepId: 'step_3',
      stepName: 'Causal Inference',
      description: 'Executing causal inference models and generating insights',
      status: 'in_progress',
      startTime: new Date(),
      metadata: { models: ['X-Learner', 'Double ML'] }
    };
    
    await updateExecutionStep(executionId, modelStep);
    
    // Execute the actual analysis (placeholder for now)
    const result = await processMacroCausalAnalysis(query);
    
    modelStep.status = 'completed';
    modelStep.endTime = new Date();
    await updateExecutionStep(executionId, modelStep);

    // Prepare the final result
    const analysisResult = removeUndefinedValues({
      success: result.success || false,
      message: result.message || "Analysis completed",
      analysis: result.analysis || null,
      metadata: result.metadata || {}
    });

    // Update status to completed with result
    await updateExecutionStatus(executionId, {
      status: 'completed',
      result: analysisResult,
      currentStep: null // Clear current step when completed
    });

    console.log(`Macro-causal analysis completed successfully: ${executionId}`);
    
  } catch (error) {
    console.error(`Error processing macro-causal analysis ${executionId}:`, error);
    
    // Update status to failed with error
    await updateExecutionStatus(executionId, {
      status: 'failed',
      error: error instanceof Error ? error.message : 'Unknown error occurred',
      currentStep: null
    });
    
    throw error; // Re-throw to trigger SQS retry mechanism
  }
}

export const handler = async (event: SQSEvent, context: Context): Promise<void> => {
  console.log(`Processing ${event.Records.length} SQS records`);

  try {
    // Process all records in parallel
    const processingPromises = event.Records.map(async (record: SQSRecord) => {
      try {
        await processAnalysisRecord(record);
      } catch (error) {
        console.error(`Failed to process record ${record.messageId}:`, error);
        // Don't throw here - let SQS handle retries
        // The error is already logged and the execution status is updated
      }
    });

    await Promise.all(processingPromises);
    console.log('All records processed successfully');

  } catch (error) {
    console.error('Error in AnalysisProcessorLambda:', error);
    // Don't throw here - let SQS handle retries
    // Individual record errors are handled above
  }
};
