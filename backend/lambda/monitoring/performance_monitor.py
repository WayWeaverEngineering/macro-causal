import os
import json
import boto3
from datetime import datetime, timedelta
from typing import Dict, Any

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function for performance monitoring.
    
    Args:
        event: Event data from EventBridge
        context: Lambda context
        
    Returns:
        Response with monitoring results
    """
    try:
        # Initialize clients
        cloudwatch = boto3.client('cloudwatch')
        sns = boto3.client('sns')
        
        # Get environment variables
        topic_arn = os.environ.get('SNS_TOPIC_ARN', '')
        environment = os.environ.get('ENVIRONMENT', 'dev')
        namespace = os.environ.get('CLOUDWATCH_NAMESPACE', 'MacroCausal')
        
        # Get metrics for the last hour
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        # Get ECS CPU utilization
        cpu_response = cloudwatch.get_metric_statistics(
            Namespace='AWS/ECS',
            MetricName='CPUUtilization',
            Dimensions=[
                {'Name': 'ClusterName', 'Value': f'macro-causal-{environment}-ml-training-cluster'},
                {'Name': 'ServiceName', 'Value': 'fastapi-service'}
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=['Average']
        )
        
        # Get ECS memory utilization
        memory_response = cloudwatch.get_metric_statistics(
            Namespace='AWS/ECS',
            MetricName='MemoryUtilization',
            Dimensions=[
                {'Name': 'ClusterName', 'Value': f'macro-causal-{environment}-ml-training-cluster'},
                {'Name': 'ServiceName', 'Value': 'fastapi-service'}
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=['Average']
        )
        
        # Analyze performance
        alerts = []
        
        if cpu_response['Datapoints']:
            avg_cpu = sum(dp['Average'] for dp in cpu_response['Datapoints']) / len(cpu_response['Datapoints'])
            if avg_cpu > 80:
                alerts.append(f"High CPU utilization: {avg_cpu:.2f}%")
        
        if memory_response['Datapoints']:
            avg_memory = sum(dp['Average'] for dp in memory_response['Datapoints']) / len(memory_response['Datapoints'])
            if avg_memory > 85:
                alerts.append(f"High memory utilization: {avg_memory:.2f}%")
        
        # Send alerts if any
        if alerts:
            alert_message = {
                'text': f"🚨 Performance Alert\n"
                        f"Environment: {environment}\n"
                        f"Time: {datetime.utcnow().isoformat()}\n"
                        f"Issues:\n" + "\n".join(f"- {alert}" for alert in alerts)
            }
            
            sns.publish(
                TopicArn=topic_arn,
                Message=json.dumps(alert_message),
                Subject=f"Performance Alert: {environment}"
            )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Performance monitoring completed',
                'alerts': alerts,
                'timestamp': datetime.utcnow().isoformat()
            })
        }
        
    except Exception as e:
        print(f"Error in performance monitoring: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }
