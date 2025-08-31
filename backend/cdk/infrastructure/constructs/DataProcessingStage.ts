import { Construct } from "constructs";
import * as path from 'path';
import { DataLakeStack } from "../stacks/DataLakeStack";
import { DefaultIdBuilder } from "../../utils/Naming";
import { EmrClusterConstruct } from "./EmrClusterConstruct";
import { Code as LambdaCode, Function as LambdaFunction, ILayerVersion } from "aws-cdk-lib/aws-lambda"
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Duration } from 'aws-cdk-lib';
import { DEFAULT_LAMBDA_NODEJS_RUNTIME } from "@wayweaver/ariadne";
import { LambdaConfig } from "../configs/LambdaConfig";

export interface DataProcessingStageProps {
  dataLakeStack: DataLakeStack;
  commonUtilsLambdaLayer: ILayerVersion;
  emrServerlessLambdaLayer: ILayerVersion;
}

export class DataProcessingStage extends Construct {
  readonly workflow: sfn.Chain;

  constructor(scope: Construct, id: string, props: DataProcessingStageProps) {
    super(scope, id);

    const dataProcessingStageName = 'data-processing';
    const dataProcessingEmrId = DefaultIdBuilder.build(`${dataProcessingStageName}-emr`);
    const emrCluster = new EmrClusterConstruct(
      this, dataProcessingEmrId, {
      name: dataProcessingStageName,
      bronzeBucket: props.dataLakeStack.bronzeBucket,
      silverBucket: props.dataLakeStack.silverBucket,
      goldBucket: props.dataLakeStack.goldBucket
    });

    // Create Lambda function to start EMR Serverless job
    const startJobLambdaId = DefaultIdBuilder.build('start-emr-job-lambda');
    const startJobLambda = new LambdaFunction(this, startJobLambdaId, {
      code: LambdaCode.fromAsset(path.join(__dirname, LambdaConfig.LAMBDA_CODE_RELATIVE_PATH)),
      handler: 'StartEmrJob.handler',
      runtime: DEFAULT_LAMBDA_NODEJS_RUNTIME,
      timeout: Duration.minutes(5),
      environment: {
        EMR_APPLICATION_ID: emrCluster.application.attrApplicationId,
        EMR_EXECUTION_ROLE_ARN: emrCluster.executionRole.roleArn,
        BRONZE_BUCKET: props.dataLakeStack.bronzeBucket.bucketName,
        SILVER_BUCKET: props.dataLakeStack.silverBucket.bucketName,
        GOLD_BUCKET: props.dataLakeStack.goldBucket.bucketName
      },
      layers: [
        props.commonUtilsLambdaLayer,
        props.emrServerlessLambdaLayer
      ]
    });

    // Create Lambda function to check EMR job status
    const checkStatusLambdaId = DefaultIdBuilder.build('check-emr-status-lambda');
    const checkStatusLambda = new LambdaFunction(this, checkStatusLambdaId, {
      code: LambdaCode.fromAsset(path.join(__dirname, LambdaConfig.LAMBDA_CODE_RELATIVE_PATH)),
      handler: 'CheckEmrStatus.handler',
      runtime: DEFAULT_LAMBDA_NODEJS_RUNTIME,
      timeout: Duration.minutes(1),
      environment: {
        EMR_APPLICATION_ID: emrCluster.application.attrApplicationId
      },
      layers: [
        props.commonUtilsLambdaLayer,
        props.emrServerlessLambdaLayer
      ]
    });

    // Grant EMR Serverless permissions to Lambda functions
    [startJobLambda, checkStatusLambda].forEach(lambdaFunc => {
      lambdaFunc.addToRolePolicy(new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'emr-serverless:StartJobRun',
          'emr-serverless:GetJobRun',
          'emr-serverless:ListJobRuns'
        ],
        resources: [emrCluster.application.attrArn]
      }));
    });

    // Grant S3 access to Lambda for reading processing scripts
    props.dataLakeStack.bronzeBucket.grantRead(startJobLambda);

    // Task to start EMR job
    const startJobTaskId = DefaultIdBuilder.build(`${dataProcessingStageName}-start-job`);
    const startJobTask = new tasks.LambdaInvoke(this, startJobTaskId, {
      stateName: `${dataProcessingStageName}-start-job`,
      lambdaFunction: startJobLambda,
      payload: sfn.TaskInput.fromObject({
        executionId: sfn.JsonPath.stringAt('$$.Execution.Id'),
        executionStartTime: sfn.JsonPath.stringAt('$$.Execution.StartTime')
      }),
      resultPath: '$.jobInfo'
    });

    // Task to check job status
    const checkStatusTaskId = DefaultIdBuilder.build(`${dataProcessingStageName}-check-status`);
    const checkStatusTask = new tasks.LambdaInvoke(this, checkStatusTaskId, {
      stateName: `${dataProcessingStageName}-check-status`,
      lambdaFunction: checkStatusLambda,
      payload: sfn.TaskInput.fromObject({
        jobRunId: sfn.JsonPath.stringAt('$.jobInfo.Payload.jobRunId')
      }),
      resultPath: '$.jobStatus'
    });

    // Wait state before checking status
    const waitState = new sfn.Wait(this, DefaultIdBuilder.build(`${dataProcessingStageName}-wait`), {
      time: sfn.WaitTime.duration(Duration.seconds(30))
    });

    // Create success and failure states
    const successState = new sfn.Succeed(this, `${dataProcessingStageName}-success`, {
      comment: 'Data processing completed successfully'
    });

    const failureState = new sfn.Fail(this, `${dataProcessingStageName}-failed`, {
      error: 'DataProcessingFailed',
      cause: 'EMR job failed',
      comment: 'Data processing stage encountered an error'
    });

    // Build the job monitoring workflow
    const dataProcessingTask = startJobTask
      .next(checkStatusTask)
      .next(new sfn.Choice(this, DefaultIdBuilder.build(`${dataProcessingStageName}-job-complete`))
        .when(sfn.Condition.stringEquals('$.jobStatus.Payload.status', 'SUCCESS'), successState)
        .when(sfn.Condition.stringEquals('$.jobStatus.Payload.status', 'FAILED'), failureState)
        .otherwise(waitState.next(checkStatusTask).next(new sfn.Choice(this, DefaultIdBuilder.build(`${dataProcessingStageName}-job-complete-2`))
          .when(sfn.Condition.stringEquals('$.jobStatus.Payload.status', 'SUCCESS'), successState)
          .when(sfn.Condition.stringEquals('$.jobStatus.Payload.status', 'FAILED'), failureState)
          .otherwise(waitState.next(checkStatusTask).next(successState))))); // Fallback to success after retries

    this.workflow = sfn.Chain.start(dataProcessingTask);
  }
}