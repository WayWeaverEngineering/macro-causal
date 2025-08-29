import os
import json
import boto3
from datetime import datetime
from typing import Dict, Any

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function for model registry operations.
    
    Args:
        event: Event data from Step Functions or EventBridge
        context: Lambda context
        
    Returns:
        Response with operation details
    """
    try:
        # Initialize clients
        dynamodb = boto3.resource('dynamodb')
        s3_client = boto3.client('s3')
        
        # Get environment variables
        table_name = os.environ.get('DYNAMODB_TABLE', '')
        s3_bucket = os.environ.get('S3_BUCKET', '')
        environment = os.environ.get('ENVIRONMENT', 'dev')
        
        table = dynamodb.Table(table_name)
        
        # Extract data from event
        model_id = event.get('modelId', '')
        version = event.get('version', 'v1.0.0')
        status = event.get('status', 'TRAINED')
        
        # Create registry entry
        registry_item = {
            'modelId': model_id,
            'version': version,
            'status': status,
            'environment': environment,
            'createdAt': datetime.utcnow().isoformat(),
            'updatedAt': datetime.utcnow().isoformat(),
            'metadata': {
                'trainingDate': datetime.utcnow().isoformat(),
                'environment': environment,
                'modelType': 'macro-causal'
            }
        }
        
        # Add TTL (1 year from now)
        registry_item['ttl'] = int(datetime.utcnow().timestamp()) + (365 * 24 * 60 * 60)
        
        # Store in DynamoDB
        table.put_item(Item=registry_item)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Model registered successfully',
                'modelId': model_id,
                'version': version,
                'status': status,
                'timestamp': datetime.utcnow().isoformat()
            })
        }
        
    except Exception as e:
        print(f"Error registering model: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }
