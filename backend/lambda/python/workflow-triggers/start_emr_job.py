import os
import json
import boto3
from datetime import datetime
from typing import Dict, Any

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function to start EMR Serverless job.
    
    Args:
        event: Event data from Step Functions
        context: Lambda context
        
    Returns:
        Response with job run details
    """
    try:
        # Initialize EMR Serverless client
        emr_client = boto3.client('emr-serverless')
        
        # Get environment variables
        environment = os.environ.get('ENVIRONMENT', 'dev')
        bronze_bucket = os.environ.get('BRONZE_BUCKET', '')
        emr_application_id = os.environ.get('EMR_APPLICATION_ID', '')
        emr_role_arn = os.environ.get('EMR_ROLE_ARN', '')
        
        # Extract data from event
        date = event.get('date', datetime.utcnow().isoformat())
        source = event.get('source', 'unknown')
        bucket = event.get('bucket', bronze_bucket)
        
        # Start EMR Serverless job
        response = emr_client.start_job_run(
            applicationId=emr_application_id,
            executionRoleArn=emr_role_arn,
            jobDriver={
                'sparkSubmit': {
                    'entryPoint': f's3://{bucket}/scripts/ingest_data.py',
                    'entryPointArguments': [f'--date={date}', f'--source={source}'],
                    'sparkSubmitParameters': ' '.join([
                        '--conf', 'spark.executor.cores=4',
                        '--conf', 'spark.executor.memory=16g',
                        '--conf', 'spark.driver.cores=4',
                        '--conf', 'spark.driver.memory=16g',
                        '--conf', 'spark.sql.adaptive.enabled=true',
                        '--conf', 'spark.sql.adaptive.coalescePartitions.enabled=true'
                    ])
                }
            },
            configurationOverrides={
                'monitoringConfiguration': {
                    'managedPersistenceMonitoringConfiguration': {
                        's3MonitoringConfiguration': {
                            'logUri': f's3://{bucket}/logs/emr/'
                        }
                    }
                }
            }
        )
        
        return {
            'statusCode': 200,
            'jobRunId': response['jobRunId'],
            'applicationId': emr_application_id,
            'state': 'STARTING',
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"Error starting EMR job: {str(e)}")
        return {
            'statusCode': 500,
            'error': str(e),
            'state': 'FAILED',
            'timestamp': datetime.utcnow().isoformat()
        }
