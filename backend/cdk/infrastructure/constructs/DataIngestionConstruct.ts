import { Construct } from 'constructs';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Duration } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS } from '../../utils/Constants';

export interface DataIngestionProps {
  bronzeBucket: any; // s3.Bucket
  emrApplication: any; // emrserverless.CfnApplication
  emrRole: any; // iam.Role
  vpc: ec2.IVpc;
  securityGroup: ec2.ISecurityGroup;
}

export class DataIngestionConstruct extends Construct {
  public readonly ingestionStateMachine: sfn.StateMachine;
  public readonly dataArrivalRule: events.Rule;
  public readonly scheduledIngestionRule: events.Rule;

  constructor(scope: Construct, id: string, props: DataIngestionProps) {
    super(scope, id);

    // Common Lambda configuration
    const lambdaConfig: any = {
      runtime: lambda.Runtime.PYTHON_3_10,
      timeout: Duration.minutes(1),
      memorySize: 512
    };

    // Lambda function to start ingestion workflow
    const ingestionTrigger = new lambda.Function(this, 'IngestionTrigger', {
      ...lambdaConfig,
      handler: 'index.handler',
      code: lambda.Code.fromAsset('../lambda/workflow-triggers'),
      environment: {
        BRONZE_BUCKET: props.bronzeBucket.bucketName,
        STATE_MACHINE_ARN: '' // Will be set after state machine creation
      },
      vpc: props.vpc,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS
      },
      securityGroups: [props.securityGroup]
    });

    // Lambda function to start EMR job
    const startEmrJob = new lambda.Function(this, 'StartEmrJob', {
      ...lambdaConfig,
      timeout: Duration.minutes(5),
      memorySize: 1024,
      handler: 'start_emr_job.handler',
      code: lambda.Code.fromAsset('../lambda/workflow-triggers'),
      environment: {
        BRONZE_BUCKET: props.bronzeBucket.bucketName,
        EMR_APPLICATION_ID: props.emrApplication.attrApplicationId,
        EMR_ROLE_ARN: props.emrRole.roleArn
      },
      vpc: props.vpc,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS
      },
      securityGroups: [props.securityGroup]
    });

    // Lambda function to check EMR job status
    const checkEmrJobStatus = new lambda.Function(this, 'CheckEmrJobStatus', {
      ...lambdaConfig,
      timeout: Duration.minutes(2),
      handler: 'check_emr_job.handler',
      code: lambda.Code.fromAsset('../lambda/workflow-triggers'),
      environment: {
        EMR_APPLICATION_ID: props.emrApplication.attrApplicationId
      },
      vpc: props.vpc,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS
      },
      securityGroups: [props.securityGroup]
    });

    // Step Functions state machine for ingestion workflow
    const dataIngestionWorkflowId = DefaultIdBuilder.build('data-ingestion-workflow');
    this.ingestionStateMachine = new sfn.StateMachine(this, dataIngestionWorkflowId, {
      definition: sfn.Chain.start(new tasks.LambdaInvoke(this, 'StartEmrJobTask', {
        lambdaFunction: startEmrJob,
        outputPath: '$.Payload'
      }))
        .next(new sfn.Wait(this, 'WaitForCompletion', {
          time: sfn.WaitTime.duration(Duration.minutes(5))
        }))
        .next(new tasks.LambdaInvoke(this, 'CheckJobStatusTask', {
          lambdaFunction: checkEmrJobStatus,
          payload: sfn.TaskInput.fromObject({
            jobRunId: sfn.JsonPath.stringAt('$.jobRunId')
          }),
          outputPath: '$.Payload'
        }))
        .next(new sfn.Choice(this, 'JobCompleted?')
          .when(sfn.Condition.stringEquals('$.state', 'SUCCESS'), 
            new sfn.Succeed(this, 'IngestionSucceeded'))
          .when(sfn.Condition.stringEquals('$.state', 'FAILED'),
            new sfn.Fail(this, 'IngestionFailed', {
              cause: 'EMR job failed',
              error: sfn.JsonPath.stringAt('$.failureReason')
            }))
          .otherwise(new sfn.Wait(this, 'WaitAndRetry', {
            time: sfn.WaitTime.duration(Duration.minutes(2))
          }).next(new tasks.LambdaInvoke(this, 'CheckJobStatusAgainTask', {
            lambdaFunction: checkEmrJobStatus,
            payload: sfn.TaskInput.fromObject({
              jobRunId: sfn.JsonPath.stringAt('$.jobRunId')
            }),
            outputPath: '$.Payload'
          })))
        ),
      timeout: Duration.minutes(MACRO_CAUSAL_CONSTANTS.STEP_FUNCTIONS.EXECUTION_TIMEOUT_MINUTES),
      stateMachineName: dataIngestionWorkflowId
    });

    // EventBridge rule for data arrival (S3 object created)
    const dataArrivalRuleId = DefaultIdBuilder.build('data-arrival-rule');
    this.dataArrivalRule = new events.Rule(this, dataArrivalRuleId, {
      ruleName: dataArrivalRuleId,
      eventPattern: {
        source: ['aws.s3'],
        detailType: ['Object Created:Put'],
        detail: {
          bucket: { name: [props.bronzeBucket.bucketName] },
          object: { key: [{ prefix: 'raw/' }] }
        }
      },
      targets: [
        new targets.SfnStateMachine(this.ingestionStateMachine, {
          input: events.RuleTargetInput.fromObject({
            date: events.EventField.fromPath('$.time'),
            source: events.EventField.fromPath('$.detail.object.key'),
            bucket: events.EventField.fromPath('$.detail.bucket.name')
          })
        })
      ]
    });

    // EventBridge rule for scheduled ingestion (daily batch)
    const scheduledIngestionRuleId = DefaultIdBuilder.build('scheduled-ingestion-rule');
    this.scheduledIngestionRule = new events.Rule(this, scheduledIngestionRuleId, {
      ruleName: scheduledIngestionRuleId,
      schedule: events.Schedule.cron({
        minute: '0',
        hour: '6', // 6 AM UTC
        day: '*',
        month: '*',
        year: '*'
      }),
      targets: [
        new targets.SfnStateMachine(this.ingestionStateMachine, {
          input: events.RuleTargetInput.fromObject({
            date: events.EventField.fromPath('$.time'),
            source: 'scheduled',
            bucket: props.bronzeBucket.bucketName
          })
        })
      ]
    });

    // Grant permissions
    props.bronzeBucket.grantReadWrite(ingestionTrigger);
    props.bronzeBucket.grantReadWrite(startEmrJob);
    props.bronzeBucket.grantReadWrite(checkEmrJobStatus);
    
    // Grant EMR permissions
    startEmrJob.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'emr-serverless:StartJobRun',
        'emr-serverless:GetJobRun'
      ],
      resources: ['*']
    }));
    
    checkEmrJobStatus.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'emr-serverless:GetJobRun'
      ],
      resources: ['*']
    }));
    
    // Grant Step Functions permissions
    this.ingestionStateMachine.grantStartExecution(ingestionTrigger);
    
    // Set the state machine ARN in the Lambda environment
    ingestionTrigger.addEnvironment('STATE_MACHINE_ARN', this.ingestionStateMachine.stateMachineArn);
  }
}
