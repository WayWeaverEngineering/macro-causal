import os
import json
import boto3
from datetime import datetime
from typing import Dict, Any

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function for drift detection alerts.
    
    Args:
        event: Event data from Evidently
        context: Lambda context
        
    Returns:
        Response with alert details
    """
    try:
        # Initialize SNS client
        sns_client = boto3.client('sns')
        
        # Get environment variables
        topic_arn = os.environ.get('SNS_TOPIC_ARN', '')
        environment = os.environ.get('ENVIRONMENT', 'dev')
        
        # Extract drift data from event
        drift_data = event.get('detail', {})
        model_id = drift_data.get('modelId', 'unknown')
        drift_score = drift_data.get('driftScore', 0.0)
        feature_name = drift_data.get('featureName', 'unknown')
        
        # Create alert message
        alert_message = {
            'text': f"🚨 Model Drift Alert\n"
                    f"Environment: {environment}\n"
                    f"Model: {model_id}\n"
                    f"Feature: {feature_name}\n"
                    f"Drift Score: {drift_score:.3f}\n"
                    f"Severity: {'HIGH' if drift_score > 0.8 else 'MEDIUM'}\n"
                    f"Timestamp: {datetime.utcnow().isoformat()}"
        }
        
        # Send to SNS
        response = sns_client.publish(
            TopicArn=topic_arn,
            Message=json.dumps(alert_message),
            Subject=f"Model Drift Alert: {model_id} - {environment}"
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Drift alert sent successfully',
                'modelId': model_id,
                'driftScore': drift_score,
                'messageId': response.get('MessageId', ''),
                'timestamp': datetime.utcnow().isoformat()
            })
        }
        
    except Exception as e:
        print(f"Error sending drift alert: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }
