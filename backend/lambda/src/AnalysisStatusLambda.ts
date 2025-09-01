import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import { handleLambdaError, JsonContentType, LambdaApiGatewayCorsHeaders, loadEnvVars } from "@wayweaver/lambda-utils";
import { GetCommand } from '@aws-sdk/lib-dynamodb';
import { AnalysisStatusResponse } from "./models/AnalysisModels";
import { DbClient } from "./clients/DbClient";

// Environment variables
const {
  ANALYSIS_EXECUTIONS_TABLE
} = loadEnvVars([
  'ANALYSIS_EXECUTIONS_TABLE'
]);

// Initialize AWS services
const dynamoDBDocumentClient = DbClient.getInstance();

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
  try {
    // Extract execution ID from path parameters
    const executionId = event.pathParameters?.executionId;
    
    if (!executionId) {
      return {
        statusCode: 400,
        headers: {
          ...JsonContentType,
          ...LambdaApiGatewayCorsHeaders
        },
        body: JSON.stringify({
          success: false,
          message: "Execution ID is required"
        })
      };
    }

    // Query DynamoDB for the execution
    const result = await dynamoDBDocumentClient.send(new GetCommand({
      TableName: ANALYSIS_EXECUTIONS_TABLE,
      Key: { executionId }
    }));

    if (!result.Item) {
      return {
        statusCode: 404,
        headers: {
          ...JsonContentType,
          ...LambdaApiGatewayCorsHeaders
        },
        body: JSON.stringify({
          success: false,
          message: "Execution not found"
        })
      };
    }

    const execution = result.Item;

    console.log(`AnalysisStatusLambda execution: ${JSON.stringify(execution)}`);
    console.log(`AnalysisStatusLambda steps:`, execution.steps);
    console.log(`AnalysisStatusLambda currentStep:`, execution.currentStep);

    // Prepare response
    const response: AnalysisStatusResponse = {
      success: true,
      message: "Execution status retrieved successfully",
      executionId: execution.executionId,
      status: execution.status,
      userQuery: execution.userQuery,
      sessionId: execution.sessionId,
      steps: execution.steps || [],
      currentStep: execution.currentStep,
      result: execution.result,
      error: execution.error,
      createdAt: execution.createdAt,
      updatedAt: execution.updatedAt
    };

    console.log(`AnalysisStatusLambda response: ${JSON.stringify(response)}`);

    return {
      statusCode: 200,
      headers: {
        ...JsonContentType,
        ...LambdaApiGatewayCorsHeaders
      },
      body: JSON.stringify(response)
    };

  } catch (error) {
    console.error('Error in AnalysisStatusLambda:', error);
    return handleLambdaError('AnalysisStatusLambda', error);
  }
};
