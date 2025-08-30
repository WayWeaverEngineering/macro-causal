import { Construct } from 'constructs';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Duration } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS } from '../../utils/Constants';
import * as actions from 'aws-cdk-lib/aws-cloudwatch-actions';
import * as ec2 from 'aws-cdk-lib/aws-ec2';

export interface MonitoringProps {
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
    const alertTopicId = DefaultIdBuilder.build('alerts');
    this.alertTopic = new sns.Topic(this, alertTopicId, {
      topicName: alertTopicId,
      displayName: 'Macro Causal Alerts'
    });

    // Lambda function for drift detection alerts
    this.driftAlert = new lambda.Function(this, 'DriftAlert', {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'drift_alert.handler',
      code: lambda.Code.fromAsset('../lambda/monitoring'),
      environment: {
        SNS_TOPIC_ARN: this.alertTopic.topicArn,
      },
      timeout: Duration.minutes(1),
      memorySize: 512,
    });

    // Lambda function for performance monitoring
    this.performanceMonitor = new lambda.Function(this, 'PerformanceMonitor', {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'performance_monitor.handler',
      code: lambda.Code.fromAsset('../lambda/monitoring'),
      environment: {
        SNS_TOPIC_ARN: this.alertTopic.topicArn,
        CLOUDWATCH_NAMESPACE: MACRO_CAUSAL_CONSTANTS.CLOUDWATCH.NAMESPACE
      },
      timeout: Duration.minutes(5),
      memorySize: 1024,
    });

    // Lambda function for data quality monitoring
    this.dataQualityMonitor = new lambda.Function(this, 'DataQualityMonitor', {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'data_quality.handler',
      code: lambda.Code.fromAsset('../lambda/monitoring'),
      environment: {
        SNS_TOPIC_ARN: this.alertTopic.topicArn,
      },
      timeout: Duration.minutes(5),
      memorySize: 1024,
    });

    // Lambda function for registry health monitoring
    this.registryHealthMonitor = new lambda.Function(this, 'RegistryHealthMonitor', {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'registry_health.handler',
      code: lambda.Code.fromAsset('../lambda/monitoring'),
      environment: {
        SNS_TOPIC_ARN: this.alertTopic.topicArn,
        DYNAMODB_TABLE: props.registryTable.tableName
      },
      timeout: Duration.minutes(2),
      memorySize: 512,
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
    const monitoringDashboardId = DefaultIdBuilder.build('monitoring-dashboard');
    const dashboard = new cloudwatch.Dashboard(this, monitoringDashboardId, {
      dashboardName: monitoringDashboardId
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
    const highCpuAlarmId = DefaultIdBuilder.build('high-cpu-alarm');
    const highCpuAlarm = new cloudwatch.Alarm(this, highCpuAlarmId, {
      metric: ecsCpuMetric,
      threshold: 80,
      evaluationPeriods: 2,
      alarmDescription: 'High CPU utilization on ECS service',
      alarmName: highCpuAlarmId
    });

    const highMemoryAlarmId = DefaultIdBuilder.build('high-memory-alarm');
    const highMemoryAlarm = new cloudwatch.Alarm(this, highMemoryAlarmId, {
      metric: ecsMemoryMetric,
      threshold: 85,
      evaluationPeriods: 2,
      alarmDescription: 'High memory utilization on ECS service',
      alarmName: highMemoryAlarmId
    });

    const highResponseTimeAlarmId = DefaultIdBuilder.build('high-response-time-alarm');
    const highResponseTimeAlarm = new cloudwatch.Alarm(this, highResponseTimeAlarmId, {
      metric: albTargetResponseTimeMetric,
      threshold: 5000, // 5 seconds
      evaluationPeriods: 2,
      alarmDescription: 'High response time on ALB',
      alarmName: highResponseTimeAlarmId
    });

    // Add alarms to SNS topic
    highCpuAlarm.addAlarmAction(new actions.SnsAction(this.alertTopic));
    highMemoryAlarm.addAlarmAction(new actions.SnsAction(this.alertTopic));
    highResponseTimeAlarm.addAlarmAction(new actions.SnsAction(this.alertTopic));

    // EventBridge rules for monitoring
    const performanceMonitoringRuleId = DefaultIdBuilder.build('performance-monitoring-rule');
    const performanceMonitoringRule = new events.Rule(this, performanceMonitoringRuleId, {
      ruleName: performanceMonitoringRuleId,
      schedule: events.Schedule.rate(Duration.hours(1)),
      targets: [new targets.LambdaFunction(this.performanceMonitor)]
    });

    const dataQualityRuleId = DefaultIdBuilder.build('data-quality-rule');
    const dataQualityRule = new events.Rule(this, dataQualityRuleId, {
      ruleName: dataQualityRuleId,
      schedule: events.Schedule.rate(Duration.hours(6)),
      targets: [new targets.LambdaFunction(this.dataQualityMonitor)]
    });

    const registryHealthRuleId = DefaultIdBuilder.build('registry-health-rule');
    const registryHealthRule = new events.Rule(this, registryHealthRuleId, {
      ruleName: registryHealthRuleId,
      schedule: events.Schedule.rate(Duration.days(1)),
      targets: [new targets.LambdaFunction(this.registryHealthMonitor)]
    });

    // Drift detection rule (triggered by Evidently)
    const driftDetectionRuleId = DefaultIdBuilder.build('drift-detection-rule');
    const driftDetectionRule = new events.Rule(this, driftDetectionRuleId, {
      ruleName: driftDetectionRuleId,
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
