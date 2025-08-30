import { Construct } from 'constructs';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as servicediscovery from 'aws-cdk-lib/aws-servicediscovery';
import { Duration } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS, RESOURCE_NAMES } from '../../utils/Constants';

export interface InferenceProps {
  environment: string;
  accountId: string;
  region: string;
  vpc: ec2.IVpc;
  artifactsBucket: s3.Bucket;
  registryTable: dynamodb.Table;
}

export class InferenceConstruct extends Construct {
  public readonly ecsCluster: ecs.Cluster;
  public readonly alb: elbv2.ApplicationLoadBalancer;
  public readonly fastAPIService: ecs.FargateService;
  public readonly inferenceRole: iam.Role;

  constructor(scope: Construct, id: string, props: InferenceProps) {
    super(scope, id);

    // ECS cluster
    this.ecsCluster = new ecs.Cluster(this, RESOURCE_NAMES.ECS_CLUSTER, {
      vpc: props.vpc,
      clusterName: DefaultIdBuilder.build('inference-cluster'),
      containerInsights: true,
      enableFargateCapacityProviders: true,
      defaultCloudMapNamespace: {
        name: 'inference',
        type: servicediscovery.NamespaceType.DNS_PRIVATE
      }
    });

    // Application Load Balancer
    this.alb = new elbv2.ApplicationLoadBalancer(this, RESOURCE_NAMES.ALB, {
      vpc: props.vpc,
      internetFacing: true,
      loadBalancerName: DefaultIdBuilder.build('inference-alb'),
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC }
    });

    // Target groups for blue/green deployment
    const blueTargetGroup = new elbv2.ApplicationTargetGroup(this, 'BlueTargetGroup', {
      vpc: props.vpc,
      port: 8000,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targetType: elbv2.TargetType.IP,
      healthCheck: {
        path: '/health',
        healthyHttpCodes: '200',
        interval: Duration.seconds(30),
        timeout: Duration.seconds(5),
        healthyThresholdCount: 2,
        unhealthyThresholdCount: 3
      }
    });

    const greenTargetGroup = new elbv2.ApplicationTargetGroup(this, 'GreenTargetGroup', {
      vpc: props.vpc,
      port: 8000,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targetType: elbv2.TargetType.IP,
      healthCheck: {
        path: '/health',
        healthyHttpCodes: '200',
        interval: Duration.seconds(30),
        timeout: Duration.seconds(5),
        healthyThresholdCount: 2,
        unhealthyThresholdCount: 3
      }
    });

    // ALB listener
    const listener = this.alb.addListener('Listener', {
      port: 80,
      protocol: elbv2.ApplicationProtocol.HTTP,
      defaultTargetGroups: [blueTargetGroup]
    });

    // IAM role for ECS tasks
    this.inferenceRole = new iam.Role(this, 'InferenceTaskRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonECSTaskExecutionRolePolicy')
      ]
    });

    // Grant access to S3 and DynamoDB
    props.artifactsBucket.grantRead(this.inferenceRole);
    props.registryTable.grantReadData(this.inferenceRole);
    
    // Add S3 permissions for model artifacts
    this.inferenceRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        's3:GetObject',
        's3:ListBucket'
      ],
      resources: [
        props.artifactsBucket.bucketArn,
        `${props.artifactsBucket.bucketArn}/*`
      ]
    }));

    // Add custom policies for inference
    this.inferenceRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'logs:CreateLogGroup',
        'logs:CreateLogStream',
        'logs:PutLogEvents',
        'logs:DescribeLogGroups',
        'logs:DescribeLogStreams'
      ],
      resources: ['*']
    }));

    // Task definition for FastAPI service
    const taskDefinition = new ecs.FargateTaskDefinition(this, 'FastAPITaskDefinition', {
      memoryLimitMiB: MACRO_CAUSAL_CONSTANTS.ECS.MEMORY_MIB,
      cpu: MACRO_CAUSAL_CONSTANTS.ECS.CPU,
      executionRole: this.inferenceRole,
      taskRole: this.inferenceRole
    });

    // Add FastAPI container
    const fastAPIContainer = taskDefinition.addContainer('FastAPIContainer', {
      image: ecs.ContainerImage.fromAsset('../ml'),
      containerName: 'fastapi-app',
      portMappings: [{ containerPort: 8000 }],
      environment: {
        ENVIRONMENT: props.environment,
        S3_BUCKET: props.artifactsBucket.bucketName,
        DYNAMODB_TABLE: props.registryTable.tableName,
        LOG_LEVEL: 'INFO'
      },
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: 'fastapi',
        logRetention: logs.RetentionDays.ONE_MONTH
      }),
      healthCheck: {
        command: ['CMD-SHELL', 'curl -f http://localhost:8000/health || exit 1'],
        interval: Duration.seconds(30),
        timeout: Duration.seconds(5),
        retries: 3,
        startPeriod: Duration.seconds(60)
      }
    });

    // Fargate service
    this.fastAPIService = new ecs.FargateService(this, 'FastAPIService', {
      cluster: this.ecsCluster,
      taskDefinition: taskDefinition,
      serviceName: DefaultIdBuilder.build('fastapi-service'),
      desiredCount: MACRO_CAUSAL_CONSTANTS.ECS.DESIRED_COUNT,
      maxHealthyPercent: 200,
      minHealthyPercent: 50,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      assignPublicIp: false,
      enableExecuteCommand: true
    });

    // Add service to target group
    this.fastAPIService.attachToApplicationTargetGroup(blueTargetGroup);

    // Auto scaling
    const scaling = this.fastAPIService.autoScaleTaskCount({
      maxCapacity: MACRO_CAUSAL_CONSTANTS.ECS.MAX_CAPACITY,
      minCapacity: 1
    });

    scaling.scaleOnCpuUtilization('CpuScaling', {
      targetUtilizationPercent: 70,
      scaleInCooldown: Duration.seconds(60),
      scaleOutCooldown: Duration.seconds(60)
    });

    scaling.scaleOnMemoryUtilization('MemoryScaling', {
      targetUtilizationPercent: 80,
      scaleInCooldown: Duration.seconds(60),
      scaleOutCooldown: Duration.seconds(60)
    });

    // Security group for ALB
    const albSecurityGroup = new ec2.SecurityGroup(this, 'ALBSecurityGroup', {
      vpc: props.vpc,
      description: 'Security group for ALB',
      allowAllOutbound: true
    });

    albSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(80),
      'Allow HTTP traffic'
    );

    // Security group for ECS tasks
    const ecsSecurityGroup = new ec2.SecurityGroup(this, 'ECSSecurityGroup', {
      vpc: props.vpc,
      description: 'Security group for ECS tasks',
      allowAllOutbound: true
    });

    ecsSecurityGroup.addIngressRule(
      albSecurityGroup,
      ec2.Port.tcp(8000),
      'Allow traffic from ALB'
    );

    // Associate security groups
    this.alb.addSecurityGroup(albSecurityGroup);
    this.fastAPIService.connections.addSecurityGroup(ecsSecurityGroup);
  }
}
