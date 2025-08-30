import os
import json
import boto3
from datetime import datetime, timedelta
from typing import Dict, Any

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function for registry health monitoring.
    
    Args:
        event: Event data from EventBridge
        context: Lambda context
        
    Returns:
        Response with registry health results
    """
    try:
        # Initialize clients
        dynamodb = boto3.resource('dynamodb')
        sns = boto3.client('sns')
        
        # Get environment variables
        topic_arn = os.environ.get('SNS_TOPIC_ARN', '')
        environment = os.environ.get('ENVIRONMENT', 'dev')
        table_name = os.environ.get('DYNAMODB_TABLE', '')
        
        if not table_name:
            raise ValueError("DYNAMODB_TABLE environment variable not set")
        
        table = dynamodb.Table(table_name)
        alerts = []
        
        # Check registry health
        try:
            # Get recent models (last 7 days)
            seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
            
            response = table.scan(
                FilterExpression='#createdAt >= :seven_days_ago',
                ExpressionAttributeNames={
                    '#createdAt': 'createdAt'
                },
                ExpressionAttributeValues={
                    ':seven_days_ago': seven_days_ago
                }
            )
            
            models = response.get('Items', [])
            
            if not models:
                alerts.append("No models registered in the last 7 days")
            else:
                # Check for models with different statuses
                status_counts = {}
                for model in models:
                    status = model.get('status', 'UNKNOWN')
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                # Check for potential issues
                if status_counts.get('FAILED', 0) > 0:
                    alerts.append(f"Found {status_counts['FAILED']} failed models")
                
                if status_counts.get('TRAINING', 0) > 3:
                    alerts.append(f"Too many models in training state: {status_counts['TRAINING']}")
                
                # Check for models without recent updates
                stale_models = []
                for model in models:
                    updated_at = model.get('updatedAt', model.get('createdAt', ''))
                    if updated_at:
                        try:
                            update_time = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                            if datetime.now(update_time.tzinfo) - update_time > timedelta(days=3):
                                stale_models.append(model.get('modelId', 'unknown'))
                        except:
                            pass
                
                if stale_models:
                    alerts.append(f"Found {len(stale_models)} models without recent updates")
                    
        except Exception as e:
            alerts.append(f"Error checking registry: {str(e)}")
        
        # Send alerts if any
        if alerts:
            alert_message = {
                'text': f"🚨 Registry Health Alert\n"
                        f"Environment: {environment}\n"
                        f"Time: {datetime.utcnow().isoformat()}\n"
                        f"Issues:\n" + "\n".join(f"- {alert}" for alert in alerts)
            }
            
            sns.publish(
                TopicArn=topic_arn,
                Message=json.dumps(alert_message),
                Subject=f"Registry Health Alert: {environment}"
            )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Registry health monitoring completed',
                'alerts': alerts,
                'timestamp': datetime.utcnow().isoformat()
            })
        }
        
    except Exception as e:
        print(f"Error in registry health monitoring: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }
