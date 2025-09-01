import { Construct } from "constructs";
import * as path from 'path';
import { DataLakeStack } from "../stacks/DataLakeStack";
import { DefaultIdBuilder } from "../../utils/Naming";
import { EmrClusterConstruct } from "./EmrClusterConstruct";
import { Code as LambdaCode, Function as LambdaFunction, ILayerVersion } from "aws-cdk-lib/aws-lambda"
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ecrAssets from 'aws-cdk-lib/aws-ecr-assets';
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
  readonly successState: sfn.Pass;

  constructor(scope: Construct, id: string, props: DataProcessingStageProps) {
    super(scope, id);

    const dataProcessingStageName = 'data-processing';

    // Create Docker image asset for data processing code
    const dataProcessingImageId = DefaultIdBuilder.build('data-processing-image');
    const dataProcessingImage = new ecrAssets.DockerImageAsset(this, dataProcessingImageId, {
      // IMPORTANT: the image path is relative to cdk.out
      directory: `../pipeline/${dataProcessingStageName}`,
      platform: ecrAssets.Platform.LINUX_AMD64,
      buildArgs: {
        'SPARK_VERSION': '3.5.5',
        'PYTHON_VERSION': '3.10'
      }
    });

    
    const dataProcessingEmrId = DefaultIdBuilder.build(`${dataProcessingStageName}-emr`);
    const emrCluster = new EmrClusterConstruct(
      this, dataProcessingEmrId, {
      name: dataProcessingStageName,
      imageUri: dataProcessingImage.imageUri,
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
          'emr-serverless:ListJobRuns',
          'emr-serverless:TagResource'
        ],
        resources: [
          emrCluster.application.attrArn,
          `${emrCluster.application.attrArn}/jobruns/*`
        ]
      }));
    });

    // Grant IAM PassRole permission to Lambda functions for EMR execution role
    [startJobLambda, checkStatusLambda].forEach(lambdaFunc => {
      lambdaFunc.addToRolePolicy(new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'iam:PassRole'
        ],
        resources: [emrCluster.executionRole.roleArn]
      }));
    });

    // Grant S3 access to Lambda for reading processing scripts
    props.dataLakeStack.bronzeBucket.grantRead(startJobLambda);

    // Task to start EMR job
    const startJobTaskId = DefaultIdBuilder.build(`${dataProcessingStageName}-start-job`);
    const startJobTask = new tasks.LambdaInvoke(this, startJobTaskId, {
      stateName: `Start ${dataProcessingStageName} Spark Job`,
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
      stateName: `Polling ${dataProcessingStageName} Spark Job`,
      lambdaFunction: checkStatusLambda,
      payload: sfn.TaskInput.fromObject({
        jobRunId: sfn.JsonPath.stringAt('$.jobInfo.Payload.jobRunId')
      }),
      resultPath: '$.jobStatus'
    });

    // Wait state before checking status
    const waitStateId = DefaultIdBuilder.build(`${dataProcessingStageName}-wait`);
    const waitState = new sfn.Wait(this, waitStateId, {
      stateName: `Waiting for ${dataProcessingStageName} Spark Job`,
      time: sfn.WaitTime.duration(Duration.seconds(30))
    });

    this.successState = new sfn.Pass(this, `${dataProcessingStageName}-success`, {
      stateName: `${dataProcessingStageName} succeeded`,
      comment: 'Data processing stage finished successfully'
    });

    const failureState = new sfn.Fail(this, `${dataProcessingStageName}-failed`, {
      error: 'DataProcessingFailed',
      cause: 'EMR job failed',
      comment: 'Data processing stage encountered an error'
    });

    // Create a choice state to check job status
    const jobStatusChoiceId = DefaultIdBuilder.build(`${dataProcessingStageName}-job-complete-choice`);
    const jobStatusChoice = new sfn.Choice(this, jobStatusChoiceId, {
      stateName: `${dataProcessingStageName} finished?`
    }).when(sfn.Condition.stringEquals('$.jobStatus.Payload.status', 'FAILED'), failureState)
      .when(sfn.Condition.stringEquals('$.jobStatus.Payload.status', 'SUCCESS'), this.successState)
      .otherwise(waitState);

    // Connect wait state back to check status task
    waitState.next(checkStatusTask);

    // Build the job monitoring workflow
    this.workflow = startJobTask
      .next(checkStatusTask)
      .next(jobStatusChoice.afterwards());
  }
}