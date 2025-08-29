import os
import json
import boto3
from datetime import datetime, timedelta
from typing import Dict, Any

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function for data quality monitoring.
    
    Args:
        event: Event data from EventBridge
        context: Lambda context
        
    Returns:
        Response with data quality results
    """
    try:
        # Initialize clients
        s3 = boto3.client('s3')
        sns = boto3.client('sns')
        
        # Get environment variables
        topic_arn = os.environ.get('SNS_TOPIC_ARN', '')
        environment = os.environ.get('ENVIRONMENT', 'dev')
        
        # Check data quality in S3 buckets
        alerts = []
        
        # Check bronze bucket for recent data
        try:
            bronze_bucket = f'macro-causal-bronze-{environment}-{os.environ.get("AWS_ACCOUNT_ID", "unknown")}'
            response = s3.list_objects_v2(
                Bucket=bronze_bucket,
                Prefix='raw/',
                MaxKeys=10
            )
            
            if 'Contents' not in response:
                alerts.append("No data found in bronze bucket")
            else:
                # Check if data is recent (within last 24 hours)
                latest_file = max(response['Contents'], key=lambda x: x['LastModified'])
                if datetime.now(latest_file['LastModified'].tzinfo) - latest_file['LastModified'] > timedelta(hours=24):
                    alerts.append(f"Data in bronze bucket is stale (last update: {latest_file['LastModified']})")
                    
        except Exception as e:
            alerts.append(f"Error checking bronze bucket: {str(e)}")
        
        # Check gold bucket for processed data
        try:
            gold_bucket = f'macro-causal-gold-{environment}-{os.environ.get("AWS_ACCOUNT_ID", "unknown")}'
            response = s3.list_objects_v2(
                Bucket=gold_bucket,
                Prefix='features/',
                MaxKeys=10
            )
            
            if 'Contents' not in response:
                alerts.append("No processed data found in gold bucket")
                
        except Exception as e:
            alerts.append(f"Error checking gold bucket: {str(e)}")
        
        # Send alerts if any
        if alerts:
            alert_message = {
                'text': f"🚨 Data Quality Alert\n"
                        f"Environment: {environment}\n"
                        f"Time: {datetime.utcnow().isoformat()}\n"
                        f"Issues:\n" + "\n".join(f"- {alert}" for alert in alerts)
            }
            
            sns.publish(
                TopicArn=topic_arn,
                Message=json.dumps(alert_message),
                Subject=f"Data Quality Alert: {environment}"
            )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Data quality monitoring completed',
                'alerts': alerts,
                'timestamp': datetime.utcnow().isoformat()
            })
        }
        
    except Exception as e:
        print(f"Error in data quality monitoring: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }
