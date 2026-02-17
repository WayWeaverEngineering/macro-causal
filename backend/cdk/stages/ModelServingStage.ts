import { Construct } from "constructs";
import { DataLakeStack } from "../stacks/DataLakeStack";
import { ConstructIdBuilder } from '@wayweaver/ariadne';
import { EcsFargateServiceConstruct } from "../constructs/EcsFargateServiceConstruct";

import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import { Effect, PolicyStatement } from "aws-cdk-lib/aws-iam";
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as acm from 'aws-cdk-lib/aws-certificatemanager';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { AwsConfig } from "../configs/AwsConfig";

export interface ModelServingStageProps {
  idBuilder: ConstructIdBuilder;
  dataLakeStack: DataLakeStack;
  modelRegistryTable: dynamodb.Table;
}

export class ModelServingStage extends Construct implements sfn.IChainable {

  readonly id: string;
  readonly startState: sfn.State;
  readonly endStates: sfn.INextable[];

  constructor(scope: Construct, id: string, props: ModelServingStageProps) {
    super(scope, id);

    this.id = id;

    const modelServingStageName = 'model-serving';
    const modelServingServiceId = props.idBuilder.build(`${modelServingStageName}-service`);
    
    // Create persistent ECS Fargate service for model serving
    const modelServingService = new EcsFargateServiceConstruct(this, modelServingServiceId, {
      name: modelServingStageName,
      idBuilder: props.idBuilder,
      // IMPORTANT: the image path is relative to the cdk/ directory (where cdk synth is run from)
      imagePath: `../pipeline/${modelServingStageName}`,
      cpu: 2048, // Higher CPU for model inference
      memoryLimitMiB: 4096, // Higher memory for model loading
      portMappings: [{ containerPort: 8080 }], // HTTP port for inference API
      persistent: true, // Run as persistent service
      assignPublicIp: true,
      environment: {
        GOLD_BUCKET: props.dataLakeStack.goldBucket.bucketName,
        ARTIFACTS_BUCKET: props.dataLakeStack.artifactsBucket.bucketName,
        MODEL_REGISTRY_TABLE: props.modelRegistryTable.tableName,
        EXECUTION_MODE: 'standalone', // Running as persistent service
        PIPELINE_EXECUTION_ID: 'persistent-service',
        EXECUTION_START_TIME: 'continuous'
      }
    });

    // Create a public Application Load Balancer for the model serving service
    const vpc = modelServingService.service.cluster.vpc;
    const albId = props.idBuilder.build(`${modelServingStageName}-alb`);
    const alb = new elbv2.ApplicationLoadBalancer(this, albId, {
      vpc,
      internetFacing: true,
    });

    // Security: allow ALB to reach the service on container port
    const serviceSecurityGroup = modelServingService.service.connections.securityGroups[0];
    const albSecurityGroup = alb.connections.securityGroups[0];
    serviceSecurityGroup.addIngressRule(
      albSecurityGroup,
      ec2.Port.tcp(8080),
      'Allow ALB to reach Fargate service'
    );

    // HTTPS listener with certificate
    const inferenceCert = acm.Certificate.fromCertificateArn(
      this,
      props.idBuilder.build('inference-domain-certificate'),
      AwsConfig.INFERENCE_DOMAIN_CERTIFICATE_ARN
    );

    const httpsListener = alb.addListener('https-listener', {
      port: 443,
      open: true,
      certificates: [inferenceCert],
    });

    // Forward HTTPS to Fargate service (container listens on 8080)
    httpsListener.addTargets('fargate-targets', {
      port: 8080,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targets: [modelServingService.service],
      healthCheck: {
        path: '/health',
        healthyHttpCodes: '200',
      },
    });

    // HTTP -> HTTPS redirect
    const httpListener = alb.addListener('http-listener', { port: 80, open: true });
    httpListener.addAction('redirect-to-https', {
      action: elbv2.ListenerAction.redirect({ protocol: 'HTTPS', port: '443' })
    });

    // Grant permissions to read from gold bucket (for processed data)
    props.dataLakeStack.goldBucket.grantRead(modelServingService.service.taskDefinition.taskRole);
    
    // Grant permissions to read from artifacts bucket (for trained models)
    props.dataLakeStack.artifactsBucket.grantRead(modelServingService.service.taskDefinition.taskRole);
    
    // Grant permissions to read from model registry table
    const modelRegistryReadStatement = new PolicyStatement({
      effect: Effect.ALLOW,
      actions: [
        'dynamodb:GetItem',
        'dynamodb:Query',
        'dynamodb:Scan',
        'dynamodb:DescribeTable'
      ],
      resources: [props.modelRegistryTable.tableArn]
    });
    modelServingService.service.taskDefinition.taskRole.addToPrincipalPolicy(modelRegistryReadStatement);

    // Grant permissions to write inference results to gold bucket
    props.dataLakeStack.goldBucket.grantWrite(modelServingService.service.taskDefinition.taskRole);

    // Create ECS task for model serving initialization
    const modelServingTaskId = props.idBuilder.build(`${modelServingStageName}-task`);
    const modelServingTask = new tasks.EcsRunTask(this, modelServingTaskId, {
      stateName: "Deploying model serving service",
      comment: "Initialize model serving service and load models - using REQUEST_RESPONSE since service runs persistently",
      cluster: modelServingService.service.cluster,
      taskDefinition: modelServingService.service.taskDefinition,
      integrationPattern: sfn.IntegrationPattern.REQUEST_RESPONSE,
      
      // Configure task parameters
      launchTarget: new tasks.EcsFargateLaunchTarget({
        platformVersion: ecs.FargatePlatformVersion.LATEST,
      }),
      
      // Add container overrides with execution context
      containerOverrides: [{
        containerDefinition: modelServingService.service.taskDefinition.defaultContainer!,
        environment: [
          { name: 'EXECUTION_MODE', value: 'step-functions' },
          { name: 'PIPELINE_EXECUTION_ID', value: sfn.JsonPath.stringAt('$$.Execution.Id') },
          { name: 'EXECUTION_START_TIME', value: sfn.JsonPath.stringAt('$$.Execution.StartTime') }
        ]
      }],
      
      // Result handling
      resultPath: '$.modelServingResult'
    });

    this.startState = modelServingTask;
    this.endStates = [modelServingTask];
  }
}
