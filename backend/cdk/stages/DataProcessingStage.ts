import { Construct } from "constructs";
import * as path from 'path';
import { DataLakeStack } from "../stacks/DataLakeStack";
import { AWS_LAMBDA_LAYERS, ConstructIdBuilder, PrebuiltLambdaLayers, UTILS_LAMBDA_LAYERS } from '@wayweaver/ariadne';
import { EmrClusterConstruct } from "../constructs/EmrClusterConstruct";
import { Code as LambdaCode, Function as LambdaFunction } from "aws-cdk-lib/aws-lambda"
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ecrAssets from 'aws-cdk-lib/aws-ecr-assets';
import { Duration } from 'aws-cdk-lib';
import { DEFAULT_LAMBDA_NODEJS_RUNTIME } from "@wayweaver/ariadne";
import { LambdaConfig } from "../configs/LambdaConfig";

export interface DataProcessingStageProps {
  idBuilder: ConstructIdBuilder;
  dataLakeStack: DataLakeStack;
  prebuiltLambdaLayers: PrebuiltLambdaLayers;
}

export class DataProcessingStage extends Construct implements sfn.IChainable {

  readonly id: string;
  readonly startState: sfn.State;
  readonly endStates: sfn.INextable[];

  constructor(scope: Construct, id: string, props: DataProcessingStageProps) {
    super(scope, id);

    this.id = id;

    const dataProcessingStageName = 'data-processing';

    // Create Docker image asset for data processing code
    const dataProcessingImageId = props.idBuilder.build('data-processing-image');
    const dataProcessingImage = new ecrAssets.DockerImageAsset(this, dataProcessingImageId, {
      // IMPORTANT: the image path is relative to cdk.out
      directory: `../pipeline/${dataProcessingStageName}`,
      platform: ecrAssets.Platform.LINUX_AMD64,
      buildArgs: {
        'SPARK_VERSION': '3.5.5',
        'PYTHON_VERSION': '3.10'
      }
    });

    
    const dataProcessingEmrId = props.idBuilder.build(`${dataProcessingStageName}-emr`);
    const emrCluster = new EmrClusterConstruct(
      this, dataProcessingEmrId, {
      name: dataProcessingStageName,
      imageUri: dataProcessingImage.imageUri,
      bronzeBucket: props.dataLakeStack.bronzeBucket,
      silverBucket: props.dataLakeStack.silverBucket,
      goldBucket: props.dataLakeStack.goldBucket,
      idBuilder: props.idBuilder,
    });

    // Create Lambda function to start EMR Serverless job
    const startJobLambdaId = props.idBuilder.build('start-emr-job-lambda');
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
        props.prebuiltLambdaLayers.getLayer(AWS_LAMBDA_LAYERS.AWS_EMR_SERVERLESS_LAMBDA_LAYER),
        props.prebuiltLambdaLayers.getLayer(UTILS_LAMBDA_LAYERS.LAMBDA_UTILS_LAMBDA_LAYER),
      ]
    });

    // Create Lambda function to check EMR job status
    const checkStatusLambdaId = props.idBuilder.build('check-emr-status-lambda');
    const checkStatusLambda = new LambdaFunction(this, checkStatusLambdaId, {
      code: LambdaCode.fromAsset(path.join(__dirname, LambdaConfig.LAMBDA_CODE_RELATIVE_PATH)),
      handler: 'CheckEmrStatus.handler',
      runtime: DEFAULT_LAMBDA_NODEJS_RUNTIME,
      timeout: Duration.minutes(1),
      environment: {
        EMR_APPLICATION_ID: emrCluster.application.attrApplicationId
      },
      layers: [
        props.prebuiltLambdaLayers.getLayer(AWS_LAMBDA_LAYERS.AWS_EMR_SERVERLESS_LAMBDA_LAYER),
        props.prebuiltLambdaLayers.getLayer(UTILS_LAMBDA_LAYERS.LAMBDA_UTILS_LAMBDA_LAYER),
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
    const startJobTaskId = props.idBuilder.build(`${dataProcessingStageName}-start-job`);
    const startJobTask = new tasks.LambdaInvoke(this, startJobTaskId, {
      stateName: "Start data processing Spark job",
      lambdaFunction: startJobLambda,
      payload: sfn.TaskInput.fromObject({
        executionId: sfn.JsonPath.stringAt('$$.Execution.Id'),
        executionStartTime: sfn.JsonPath.stringAt('$$.Execution.StartTime')
      }),
      resultPath: '$.jobInfo'
    });

    // Task to check job status
    const checkStatusTaskId = props.idBuilder.build(`${dataProcessingStageName}-check-status`);
    const checkStatusTask = new tasks.LambdaInvoke(this, checkStatusTaskId, {
      stateName: "Polling Spark job status",
      lambdaFunction: checkStatusLambda,
      payload: sfn.TaskInput.fromObject({
        jobRunId: sfn.JsonPath.stringAt('$.jobInfo.Payload.jobRunId')
      }),
      resultPath: '$.jobStatus'
    });

    // Wait state before checking status
    const waitStateId = props.idBuilder.build(`${dataProcessingStageName}-wait`);
    const waitState = new sfn.Wait(this, waitStateId, {
      stateName: "Waiting for Spark job",
      time: sfn.WaitTime.duration(Duration.seconds(30))
    });

    // Connect wait state back to check status task
    waitState.next(checkStatusTask);

    const successStateId = props.idBuilder.build(`${dataProcessingStageName}-success`);
    const successState = new sfn.Pass(this, successStateId, {
      stateName: "Data processing stage succeeded",
      comment: 'Data processing stage finished successfully'
    });

    const failureStateId = props.idBuilder.build(`${dataProcessingStageName}-failed`);
    const failureState = new sfn.Fail(this, failureStateId, {
      stateName: "Data processing stage failed",
      comment: 'Data processing stage encountered an error'
    });

    // Create a choice state to check job status
    const jobStatusChoiceId = props.idBuilder.build(`${dataProcessingStageName}-job-complete-choice`);
    const jobStatusChoice = new sfn.Choice(this, jobStatusChoiceId, {
      stateName: "Spark job status?"
    }).when(sfn.Condition.stringEquals('$.jobStatus.Payload.status', 'FAILED'), failureState)
      .when(sfn.Condition.stringEquals('$.jobStatus.Payload.status', 'SUCCESS'), successState)
      .otherwise(waitState);

    // Build the job monitoring workflow
    startJobTask.next(checkStatusTask).next(jobStatusChoice);

    this.startState = startJobTask;
    this.endStates = [successState];
  }
}