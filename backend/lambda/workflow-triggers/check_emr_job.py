import os
import json
import boto3
from datetime import datetime
from typing import Dict, Any

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function to check EMR Serverless job status.
    
    Args:
        event: Event data from Step Functions
        context: Lambda context
        
    Returns:
        Response with job status
    """
    try:
        # Initialize EMR Serverless client
        emr_client = boto3.client('emr-serverless')
        
        # Get environment variables
        emr_application_id = os.environ.get('EMR_APPLICATION_ID', '')
        
        # Extract job run ID from event
        job_run_id = event.get('jobRunId', '')
        
        if not job_run_id:
            return {
                'statusCode': 400,
                'error': 'No jobRunId provided',
                'state': 'FAILED',
                'timestamp': datetime.utcnow().isoformat()
            }
        
        # Get job run status
        response = emr_client.get_job_run(
            applicationId=emr_application_id,
            jobRunId=job_run_id
        )
        
        job_run = response['jobRun']
        state = job_run['state']
        failure_reason = job_run.get('failureReason', '')
        
        return {
            'statusCode': 200,
            'jobRunId': job_run_id,
            'applicationId': emr_application_id,
            'state': state,
            'failureReason': failure_reason,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"Error checking EMR job status: {str(e)}")
        return {
            'statusCode': 500,
            'error': str(e),
            'state': 'FAILED',
            'timestamp': datetime.utcnow().isoformat()
        }
