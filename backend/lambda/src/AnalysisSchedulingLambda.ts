import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import { handleLambdaError, JsonContentType, LambdaApiGatewayCorsHeaders, loadEnvVars } from "@wayweaver/lambda-utils";
import { PutCommand } from '@aws-sdk/lib-dynamodb';
import { sendMessageToQueue } from "@wayweaver/aws-queue";
import { AnalysisMessage, AnalysisRequest, AnalysisScheduleResponse } from "./models/AnalysisModels";
import { DbClient } from "./clients/DbClient";

// Environment variables
const {
  ANALYSIS_EXECUTIONS_TABLE,
  ANALYSIS_EXECUTIONS_QUEUE_URL,
} = loadEnvVars([
  'ANALYSIS_EXECUTIONS_TABLE',
  'ANALYSIS_EXECUTIONS_QUEUE_URL',
]);

// Initialize AWS services
const dynamoDBDocumentClient = DbClient.getInstance();

// Helper function to generate execution ID
function generateExecutionId(): string {
  return `macro_causal_exec_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
}

export const handler: APIGatewayProxyHandlerV2 = async (event) => {

  const defaultHeaders = {
    ...JsonContentType,
    ...LambdaApiGatewayCorsHeaders
  };

  try {
    // Parse request
    const requestBody: AnalysisRequest = event.body ? JSON.parse(event.body) : {};
    const { query, sessionId } = requestBody;

    // Validate request
    if (!query || query.trim().length === 0) {
      return {
        statusCode: 400,
        headers: defaultHeaders,
        body: JSON.stringify({
          success: false,
          message: "Query is required"
        })
      };
    }

    // Generate execution ID
    const executionId = generateExecutionId();

    // Create initial DynamoDB record
    const initialRecord = {
      executionId,
      status: 'pending',
      userQuery: query.trim(),
      sessionId,
      steps: [],
      currentStep: null,
      result: null,
      error: null,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    await dynamoDBDocumentClient.send(new PutCommand({
      TableName: ANALYSIS_EXECUTIONS_TABLE,
      Item: initialRecord
    }));

    // Send message to SQS queue
    const message: AnalysisMessage = {
      sessionId,
      executionId,
      query: query.trim()
    };
    await sendMessageToQueue(ANALYSIS_EXECUTIONS_QUEUE_URL, message);

    console.log(`Macro-causal analysis scheduled successfully: ${executionId}`);

    const response: AnalysisScheduleResponse = {
      success: true,
      message: "Macro-causal analysis scheduled successfully",
      executionId
    };

    return {
      statusCode: 200,
      headers: defaultHeaders,
      body: JSON.stringify(response)
    };

  } catch (error) {
    console.error('Error in AnalysisSchedulingLambda:', error);
    return handleLambdaError('AnalysisSchedulingLambda', error);
  }
};
