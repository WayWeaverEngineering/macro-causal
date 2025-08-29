import { Construct } from 'constructs';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as subscriptions from 'aws-cdk-lib/aws-sns-subscriptions';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import { Duration } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS, RESOURCE_NAMES } from '../../utils/Constants';
import * as actions from 'aws-cdk-lib/aws-cloudwatch-actions';

export interface MonitoringProps {
  environment: string;
  accountId: string;
  region: string;
  ecsCluster: any; // ecs.Cluster
  alb: any; // elbv2.ApplicationLoadBalancer
  registryTable: any; // dynamodb.Table
}

export class MonitoringConstruct extends Construct {
  public readonly alertTopic: sns.Topic;
  public readonly driftAlert: lambda.Function;
  public readonly performanceMonitor: lambda.Function;
  public readonly dataQualityMonitor: lambda.Function;
  public readonly registryHealthMonitor: lambda.Function;

  constructor(scope: Construct, id: string, props: MonitoringProps) {
    super(scope, id);

    // SNS topic for alerts
    this.alertTopic = new sns.Topic(this, RESOURCE_NAMES.ALERT_TOPIC, {
      topicName: DefaultIdBuilder.build('alerts'),
      displayName: 'Macro Causal Alerts'
    });

    // Lambda function for drift detection alerts
    this.driftAlert = new lambda.Function(this, 'DriftAlert', {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'drift_alert.handler',
      code: lambda.Code.fromAsset('../lambda/monitoring'),
      environment: {
        SNS_TOPIC_ARN: this.alertTopic.topicArn,
        ENVIRONMENT: props.environment
      },
      timeout: Duration.minutes(1),
      memorySize: 512
    });

    // Lambda function for performance monitoring
    this.performanceMonitor = new lambda.Function(this, 'PerformanceMonitor', {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'performance_monitor.handler',
      code: lambda.Code.fromAsset('../lambda/monitoring'),
      environment: {
        SNS_TOPIC_ARN: this.alertTopic.topicArn,
        ENVIRONMENT: props.environment,
        CLOUDWATCH_NAMESPACE: MACRO_CAUSAL_CONSTANTS.CLOUDWATCH.NAMESPACE
      },
      timeout: Duration.minutes(5),
      memorySize: 1024
    });

    // Lambda function for data quality monitoring
    this.dataQualityMonitor = new lambda.Function(this, 'DataQualityMonitor', {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'data_quality.handler',
      code: lambda.Code.fromAsset('../lambda/monitoring'),
      environment: {
        SNS_TOPIC_ARN: this.alertTopic.topicArn,
        ENVIRONMENT: props.environment
      },
      timeout: Duration.minutes(5),
      memorySize: 1024
    });

    // Lambda function for registry health monitoring
    this.registryHealthMonitor = new lambda.Function(this, 'RegistryHealthMonitor', {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'registry_health.handler',
      code: lambda.Code.fromAsset('../lambda/monitoring'),
      environment: {
        SNS_TOPIC_ARN: this.alertTopic.topicArn,
        ENVIRONMENT: props.environment,
        DYNAMODB_TABLE: props.registryTable.tableName
      },
      timeout: Duration.minutes(2),
      memorySize: 512
    });

    // Grant permissions to Lambda functions
    this.alertTopic.grantPublish(this.driftAlert);
    this.alertTopic.grantPublish(this.performanceMonitor);
    this.alertTopic.grantPublish(this.dataQualityMonitor);
    this.alertTopic.grantPublish(this.registryHealthMonitor);

    // Grant CloudWatch permissions
    [this.driftAlert, this.performanceMonitor, this.dataQualityMonitor, this.registryHealthMonitor].forEach(func => {
      func.addToRolePolicy(new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'cloudwatch:GetMetricStatistics',
          'cloudwatch:PutMetricData',
          'cloudwatch:DescribeAlarms',
          'logs:CreateLogGroup',
          'logs:CreateLogStream',
          'logs:PutLogEvents'
        ],
        resources: ['*']
      }));
    });

    // Grant DynamoDB permissions to registry health monitor
    props.registryTable.grantReadData(this.registryHealthMonitor);

    // CloudWatch dashboard
    const dashboard = new cloudwatch.Dashboard(this, 'MonitoringDashboard', {
      dashboardName: DefaultIdBuilder.build('monitoring-dashboard')
    });

    // ECS metrics
    const ecsCpuMetric = new cloudwatch.Metric({
      namespace: 'AWS/ECS',
      metricName: 'CPUUtilization',
      dimensionsMap: {
        ClusterName: props.ecsCluster.clusterName,
        ServiceName: 'fastapi-service'
      },
      statistic: 'Average',
      period: Duration.minutes(5)
    });

    const ecsMemoryMetric = new cloudwatch.Metric({
      namespace: 'AWS/ECS',
      metricName: 'MemoryUtilization',
      dimensionsMap: {
        ClusterName: props.ecsCluster.clusterName,
        ServiceName: 'fastapi-service'
      },
      statistic: 'Average',
      period: Duration.minutes(5)
    });

    // ALB metrics
    const albRequestCountMetric = new cloudwatch.Metric({
      namespace: 'AWS/ApplicationELB',
      metricName: 'RequestCount',
      dimensionsMap: {
        LoadBalancer: props.alb.loadBalancerFullName,
        TargetGroup: 'targetgroup/fastapi'
      },
      statistic: 'Sum',
      period: Duration.minutes(1)
    });

    const albTargetResponseTimeMetric = new cloudwatch.Metric({
      namespace: 'AWS/ApplicationELB',
      metricName: 'TargetResponseTime',
      dimensionsMap: {
        LoadBalancer: props.alb.loadBalancerFullName,
        TargetGroup: 'targetgroup/fastapi'
      },
      statistic: 'Average',
      period: Duration.minutes(5)
    });

    // Add widgets to dashboard
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'ECS CPU Utilization',
        left: [ecsCpuMetric],
        width: 12,
        height: 6
      }),
      new cloudwatch.GraphWidget({
        title: 'ECS Memory Utilization',
        left: [ecsMemoryMetric],
        width: 12,
        height: 6
      }),
      new cloudwatch.GraphWidget({
        title: 'ALB Request Count',
        left: [albRequestCountMetric],
        width: 12,
        height: 6
      }),
      new cloudwatch.GraphWidget({
        title: 'ALB Response Time',
        left: [albTargetResponseTimeMetric],
        width: 12,
        height: 6
      })
    );

    // CloudWatch alarms
    const highCpuAlarm = new cloudwatch.Alarm(this, 'HighCPUAlarm', {
      metric: ecsCpuMetric,
      threshold: 80,
      evaluationPeriods: 2,
      alarmDescription: 'High CPU utilization on ECS service',
      alarmName: DefaultIdBuilder.build('high-cpu-alarm')
    });

    const highMemoryAlarm = new cloudwatch.Alarm(this, 'HighMemoryAlarm', {
      metric: ecsMemoryMetric,
      threshold: 85,
      evaluationPeriods: 2,
      alarmDescription: 'High memory utilization on ECS service',
      alarmName: DefaultIdBuilder.build('high-memory-alarm')
    });

    const highResponseTimeAlarm = new cloudwatch.Alarm(this, 'HighResponseTimeAlarm', {
      metric: albTargetResponseTimeMetric,
      threshold: 5000, // 5 seconds
      evaluationPeriods: 2,
      alarmDescription: 'High response time on ALB',
      alarmName: DefaultIdBuilder.build('high-response-time-alarm')
    });

    // Add alarms to SNS topic
    highCpuAlarm.addAlarmAction(new actions.SnsAction(this.alertTopic));
    highMemoryAlarm.addAlarmAction(new actions.SnsAction(this.alertTopic));
    highResponseTimeAlarm.addAlarmAction(new actions.SnsAction(this.alertTopic));

    // EventBridge rules for monitoring
    const performanceMonitoringRule = new events.Rule(this, 'PerformanceMonitoringRule', {
      ruleName: DefaultIdBuilder.build('performance-monitoring-rule'),
      schedule: events.Schedule.rate(Duration.hours(1)),
      targets: [new targets.LambdaFunction(this.performanceMonitor)]
    });

    const dataQualityRule = new events.Rule(this, 'DataQualityRule', {
      ruleName: DefaultIdBuilder.build('data-quality-rule'),
      schedule: events.Schedule.rate(Duration.hours(6)),
      targets: [new targets.LambdaFunction(this.dataQualityMonitor)]
    });

    const registryHealthRule = new events.Rule(this, 'RegistryHealthRule', {
      ruleName: DefaultIdBuilder.build('registry-health-rule'),
      schedule: events.Schedule.rate(Duration.days(1)),
      targets: [new targets.LambdaFunction(this.registryHealthMonitor)]
    });

    // Drift detection rule (triggered by Evidently)
    const driftDetectionRule = new events.Rule(this, 'DriftDetectionRule', {
      ruleName: DefaultIdBuilder.build('drift-detection-rule'),
      eventPattern: {
        source: ['aws.evidently'],
        detailType: ['Evidently Drift Detection'],
        detail: {
          driftScore: [{ numeric: ['>', 0.8] }]
        }
      },
      targets: [new targets.LambdaFunction(this.driftAlert)]
    });
  }
}
