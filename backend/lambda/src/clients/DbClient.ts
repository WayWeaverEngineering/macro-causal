import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb';

/**
 * DynamoDB Client Wrapper
 * 
 * Provides a centralized DynamoDB client instance for the analysis service.
 * This ensures consistent configuration and connection management across all operations.
 */
export class DbClient {
  private static instance: DynamoDBDocumentClient;
  
  /**
   * Get the singleton DynamoDB document client instance
   */
  public static getInstance(): DynamoDBDocumentClient {
    if (!DbClient.instance) {
      const dynamoClient = new DynamoDBClient({});
      DbClient.instance = DynamoDBDocumentClient.from(dynamoClient, {
        marshallOptions: {
          removeUndefinedValues: true
        }
      });
    }
    return DbClient.instance;
  }
  
  /**
   * Reset the client instance (useful for testing)
   */
  public static resetInstance(): void {
    DbClient.instance = undefined as any;
  }
}
