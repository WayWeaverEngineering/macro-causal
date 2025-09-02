import { SQSEvent, SQSRecord, Context } from "aws-lambda";
import { parseSqsRecord } from "@wayweaver/aws-queue";
import { loadEnvVars } from "@wayweaver/lambda-utils";
import { ExecutionStep } from "./models/AgentModels";
import { UpdateCommand, GetCommand } from '@aws-sdk/lib-dynamodb';
import { AnalysisExecution, AnalysisMessage } from "./models/AnalysisModels";
import { DbClient } from "./clients/DbClient";
import { initializeUuidPolyfill } from "./utils/UuidPolyfill";
import { OpenAIService } from "./services/OpenAIService";
import { getSecretValue } from "@wayweaver/aws-secrets";
import { ModelServingService } from "./services/ModelServingService";

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
  OPENAI_API_SECRET_ID,
} = loadEnvVars([
  'ANALYSIS_EXECUTIONS_TABLE',
  'OPENAI_API_SECRET_ID'
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

// Step 1: Convert user query to model inputs
async function convertQueryToModelInputs(
  query: string, 
  openAIService: OpenAIService
): Promise<any> {
  console.log(`Converting query to model inputs: ${query}`);
  
  const prompt = `
You are an expert macro-causal analysis assistant. Convert this user query into structured inputs for the trained models.

USER QUERY: "${query}"

Based on the query, generate the appropriate input format for the hybrid causal inference models. The models expect:

1. For causal effect analysis (X-Learner):
   - Treatment variables (macro shocks like Fed rate changes, CPI surprises, GDP shocks)
   - Outcome variables (asset returns like S&P 500, bonds, gold)
   - Confounding variables (other macro indicators)
   - Time periods for analysis

2. For regime classification:
   - Market indicators (VIX, yield curves, economic indicators)
   - Lookback periods for regime identification

3. For uncertainty estimation:
   - Base estimates to quantify uncertainty around

Generate a JSON response with the appropriate model inputs. Focus on the most relevant model type for this query.

RESPONSE FORMAT (JSON only):
{
  "model_type": "hybrid_causal_model",
  "inputs": {
    "treatment_variables": ["fed_rate_shock", "cpi_surprise"],
    "outcome_variables": ["sp500_returns", "bond_returns"],
    "confounders": ["gdp_growth", "unemployment_rate", "oil_prices"],
    "time_periods": {"start": "2020-01", "end": "2024-01"},
    "market_indicators": ["vix", "yield_curve_slope", "economic_surprise_index"],
    "lookback_periods": 12
  }
}
`;

  try {
    const response = await openAIService.getModel().invoke(prompt);
    const content = response.content as string;
    
    // Extract JSON from response
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error("No JSON found in OpenAI response");
    }
    
    const modelInputs = JSON.parse(jsonMatch[0]);
    console.log(`Generated model inputs:`, modelInputs);
    
    return modelInputs;
  } catch (error) {
    console.error(`Error converting query to model inputs:`, error);
    throw error;
  }
}

// Step 2: Submit model inference requests
async function executeModelInference(
  modelInputs: any, 
  modelServingService: ModelServingService
): Promise<any> {
  console.log(`Executing model inference with inputs:`, modelInputs);
  
  try {
    // Submit inference request to the model serving service
    const inferenceResult = await modelServingService.submitInferenceRequest(modelInputs);
    console.log(`Model inference completed:`, inferenceResult);
    
    return inferenceResult;
  } catch (error) {
    console.error(`Error executing model inference:`, error);
    throw error;
  }
}

// Step 3: Generate final response
async function generateFinalResponse(
  query: string, 
  modelResults: any, 
  openAIService: OpenAIService
): Promise<string> {
  console.log(`Generating final response for query: ${query}`);
  
  const prompt = `
You are an expert macro-causal analysis assistant. Generate a comprehensive, professional response based on the user's query and the model results.

USER QUERY: "${query}"

MODEL RESULTS:
${JSON.stringify(modelResults, null, 2)}

Generate a detailed response that:
1. Directly answers the user's question
2. Explains the causal effects found by the models
3. Interprets the regime classification and uncertainty estimates
4. Provides actionable insights and policy implications
5. Acknowledges limitations and assumptions
6. Uses professional, Bridgewater-style analytical language

Focus on:
- Causal relationships, not just correlations
- Economic significance of the effects
- Regime-dependent behavior
- Confidence levels and uncertainty
- Practical investment implications

RESPONSE:
`;

  try {
    const response = await openAIService.getModel().invoke(prompt);
    const finalResponse = response.content as string;
    console.log(`Generated final response:`, finalResponse);
    
    return finalResponse;
  } catch (error) {
    console.error(`Error generating final response:`, error);
    throw error;
  }
}

// Main analysis processing function
async function processMacroCausalAnalysis(
  query: string, 
  executionId: string
): Promise<any> {
  console.log(`Processing macro-causal analysis for query: ${query}`);
  
  try {
    // Get OpenAI API key
    const openAIApiKey = await getSecretValue(OPENAI_API_SECRET_ID, "API_KEY");
    if (!openAIApiKey) {
      throw new Error("OpenAI API key not found");
    }

    // Initialize services
    const openAIService = new OpenAIService(openAIApiKey);

    const MODEL_SERVING_URL =  "https://macro-ai-analyst-inference.wayweaver.com";
    const modelServingService = new ModelServingService(MODEL_SERVING_URL);

    // Step 1: Convert query to model inputs
    const queryStep: ExecutionStep = {
      stepId: 'step_1',
      stepName: 'Query Analysis & Input Generation',
      description: 'Converting natural language query to structured model inputs',
      status: 'in_progress',
      startTime: new Date(),
      metadata: { query }
    };
    
    await updateExecutionStep(executionId, queryStep);
    
    const modelInputs = await convertQueryToModelInputs(query, openAIService);
    
    queryStep.status = 'completed';
    queryStep.endTime = new Date();
    queryStep.metadata = { ...queryStep.metadata, modelInputs };
    await updateExecutionStep(executionId, queryStep);

    // Step 2: Execute model inference
    const modelStep: ExecutionStep = {
      stepId: 'step_2',
      stepName: 'Model Inference',
      description: 'Executing hybrid causal inference models (X-Learner + PyTorch)',
      status: 'in_progress',
      startTime: new Date(),
      metadata: { modelInputs }
    };
    
    await updateExecutionStep(executionId, modelStep);
    
    const modelResults = await executeModelInference(modelInputs, modelServingService);
    
    modelStep.status = 'completed';
    modelStep.endTime = new Date();
    modelStep.metadata = { ...modelStep.metadata, modelResults };
    await updateExecutionStep(executionId, modelStep);

    // Step 3: Generate final response
    const responseStep: ExecutionStep = {
      stepId: 'step_3',
      stepName: 'Response Generation',
      description: 'Generating comprehensive analysis response',
      status: 'in_progress',
      startTime: new Date(),
      metadata: { modelResults }
    };
    
    await updateExecutionStep(executionId, responseStep);
    
    const finalResponse = await generateFinalResponse(query, modelResults, openAIService);
    
    responseStep.status = 'completed';
    responseStep.endTime = new Date();
    responseStep.metadata = { ...responseStep.metadata, finalResponse };
    await updateExecutionStep(executionId, responseStep);

    // Prepare the final result
    const analysisResult = removeUndefinedValues({
      success: true,
      message: "Macro-causal analysis completed successfully",
      analysis: {
        insight_type: 'treatment_effect',
        summary: finalResponse,
        model_results: modelResults,
        model_inputs: modelInputs,
        methodology: "Hybrid X-Learner with Double ML + PyTorch Regime/Uncertainty"
      },
      metadata: {
        executionTime: Date.now() - queryStep.startTime!.getTime(),
        analysisType: "hybrid_causal_inference",
        complexity: "advanced",
        stepsCompleted: 3
      }
    });

    return analysisResult;
    
  } catch (error) {
    console.error(`Error in macro-causal analysis:`, error);
    throw error;
  }
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

    // Execute the full analysis pipeline
    const result = await processMacroCausalAnalysis(query, executionId);

    // Update status to completed with result
    await updateExecutionStatus(executionId, {
      status: 'completed',
      result: result,
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
