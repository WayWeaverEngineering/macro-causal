import os
import json
import boto3
from datetime import datetime
from typing import Dict, Any

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function for model promotion.
    
    Args:
        event: Event data from EventBridge
        context: Lambda context
        
    Returns:
        Response with promotion details
    """
    try:
        # Initialize clients
        dynamodb = boto3.resource('dynamodb')
        s3_client = boto3.client('s3')
        
        # Get environment variables
        table_name = os.environ.get('DYNAMODB_TABLE', '')
        s3_bucket = os.environ.get('S3_BUCKET', '')
        environment = os.environ.get('ENVIRONMENT', 'dev')
        
        if not table_name:
            raise ValueError("DYNAMODB_TABLE environment variable not set")
        
        table = dynamodb.Table(table_name)
        
        # Extract data from event
        model_id = event.get('modelId', '')
        version = event.get('version', 'v1.0.0')
        target_status = event.get('targetStatus', 'PRODUCTION')
        
        if not model_id:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'modelId is required',
                    'timestamp': datetime.utcnow().isoformat()
                })
            }
        
        # Update model status in registry
        try:
            response = table.update_item(
                Key={
                    'modelId': model_id,
                    'version': version
                },
                UpdateExpression='SET #status = :status, #updatedAt = :updatedAt',
                ExpressionAttributeNames={
                    '#status': 'status',
                    '#updatedAt': 'updatedAt'
                },
                ExpressionAttributeValues={
                    ':status': target_status,
                    ':updatedAt': datetime.utcnow().isoformat()
                },
                ReturnValues='ALL_NEW'
            )
            
            updated_item = response.get('Attributes', {})
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Model promoted successfully',
                    'modelId': model_id,
                    'version': version,
                    'newStatus': target_status,
                    'timestamp': datetime.utcnow().isoformat()
                })
            }
            
        except Exception as e:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': f'Failed to update model status: {str(e)}',
                    'timestamp': datetime.utcnow().isoformat()
                })
            }
        
    except Exception as e:
        print(f"Error promoting model: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }
