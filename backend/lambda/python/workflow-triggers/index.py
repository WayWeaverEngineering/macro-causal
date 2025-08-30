import os
import json
import boto3
from datetime import datetime
from typing import Dict, Any

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function to trigger data ingestion workflows.
    
    Args:
        event: Event data from EventBridge
        context: Lambda context
        
    Returns:
        Response with workflow execution details
    """
    try:
        # Initialize Step Functions client
        sfn_client = boto3.client('stepfunctions')
        
        # Get environment variables
        environment = os.environ.get('ENVIRONMENT', 'dev')
        bronze_bucket = os.environ.get('BRONZE_BUCKET', '')
        
        # Extract data from event
        date = event.get('date', datetime.utcnow().isoformat())
        source = event.get('source', 'unknown')
        bucket = event.get('bucket', bronze_bucket)
        
        # Start Step Functions execution
        execution_name = f"data-ingestion-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        response = sfn_client.start_execution(
            stateMachineArn=os.environ.get('STATE_MACHINE_ARN', ''),
            name=execution_name,
            input=json.dumps({
                'date': date,
                'source': source,
                'bucket': bucket,
                'environment': environment
            })
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Workflow triggered successfully',
                'executionArn': response['executionArn'],
                'executionName': execution_name,
                'timestamp': datetime.utcnow().isoformat()
            })
        }
        
    except Exception as e:
        print(f"Error triggering workflow: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }
