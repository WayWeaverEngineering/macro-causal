import { APIGatewayProxyHandlerV2 } from "aws-lambda";
import { getSecretValue } from "@wayweaver/aws-secrets";
import { MacroCausalOrchestrator } from "../orchestrators/MacroCausalOrchestrator";
import { OpenAIService } from "../services/OpenAIService";
import { ModelService } from "../services/ModelService";
import { QueryRequest } from "../models/QueryModels";
import { QueryResponse } from "../models/AgentModels";

// Environment variables
const OPENAI_API_SECRET_ID = process.env.OPENAI_API_SECRET_ID || 'openai-api-key';
const MODEL_SERVING_URL = process.env.MODEL_SERVING_URL || 'http://localhost:8000';
const MODEL_SERVING_API_KEY = process.env.MODEL_SERVING_API_KEY;

export const handler: APIGatewayProxyHandlerV2 = async (event) => {
  console.log('QueryHandler.handler - Event received:', JSON.stringify(event, null, 2));

  const defaultHeaders = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
    'Access-Control-Allow-Methods': 'POST,OPTIONS'
  };

  try {
    // Handle CORS preflight request
    if (event.requestContext?.http?.method === 'OPTIONS') {
      return {
        statusCode: 200,
        headers: defaultHeaders,
        body: JSON.stringify({ message: 'CORS preflight successful' })
      };
    }

    // Parse request
    const requestBody: QueryRequest = event.body ? JSON.parse(event.body) : {};
    const { query, sessionId, options } = requestBody;

    // Validate request
    if (!query || query.trim().length === 0) {
      console.log('QueryHandler.handler - Invalid request: missing query');
      return {
        statusCode: 400,
        headers: defaultHeaders,
        body: JSON.stringify({
          success: false,
          error: 'Query is required'
        })
      };
    }

    console.log('QueryHandler.handler - Processing query:', query);
    console.log('QueryHandler.handler - Session ID:', sessionId);
    console.log('QueryHandler.handler - Options:', options);

    // Get OpenAI API key
    const openAIApiKey = await getSecretValue(OPENAI_API_SECRET_ID, "API_KEY");
    if (!openAIApiKey) {
      console.error('QueryHandler.handler - OpenAI API key not found');
      return {
        statusCode: 500,
        headers: defaultHeaders,
        body: JSON.stringify({
          success: false,
          error: 'OpenAI API key not configured'
        })
      };
    }

    // Initialize services
    const openAIService = new OpenAIService(openAIApiKey);
    const modelService = new ModelService(MODEL_SERVING_URL, MODEL_SERVING_API_KEY);

    // Test OpenAI connection
    const openAIConnection = await openAIService.testConnection();
    if (!openAIConnection) {
      console.error('QueryHandler.handler - OpenAI connection test failed');
      return {
        statusCode: 500,
        headers: defaultHeaders,
        body: JSON.stringify({
          success: false,
          error: 'OpenAI service unavailable'
        })
      };
    }

    console.log('QueryHandler.handler - OpenAI connection test successful');

    // Initialize orchestrator
    const orchestrator = new MacroCausalOrchestrator(openAIService, modelService);

    // Execute query
    const result: QueryResponse = await orchestrator.executeQuery(query.trim());

    console.log('QueryHandler.handler - Query execution completed');
    console.log('QueryHandler.handler - Result success:', result.success);

    // Return response
    return {
      statusCode: result.success ? 200 : 400,
      headers: defaultHeaders,
      body: JSON.stringify(result)
    };

  } catch (error) {
    console.error('QueryHandler.handler - Unexpected error:', error);
    
    return {
      statusCode: 500,
      headers: defaultHeaders,
      body: JSON.stringify({
        success: false,
        error: 'Internal server error',
        message: error instanceof Error ? error.message : 'Unknown error occurred'
      })
    };
  }
};
