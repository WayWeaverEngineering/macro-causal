import { Construct } from "constructs";
import { DataLakeStack } from "../stacks/DataLakeStack";
import { DefaultIdBuilder } from "../../utils/Naming";
import { EmrClusterConstruct } from "./EmrClusterConstruct";
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';

export interface DataProcessingStageProps {
  dataLakeStack: DataLakeStack;
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

    const dataProcessingTaskId = DefaultIdBuilder.build(`${dataProcessingStageName}-task`);
    const dataProcessingTask = new tasks.EmrServerlessStartJobRun(this, dataProcessingTaskId, {
      stateName: dataProcessingStageName,
      applicationId: emrCluster.application.attrApplicationId,
      executionRoleArn: emrCluster.executionRole.roleArn,
      jobDriver: {
        sparkSubmit: {
          entryPoint: `s3://${props.dataLakeStack.bronzeBucket.bucketName}/processing-scripts/data_processing.py`,
          entryPointArguments: [
            '--bronze-bucket', props.dataLakeStack.bronzeBucket.bucketName,
            '--silver-bucket', props.dataLakeStack.silverBucket.bucketName,
            '--gold-bucket', props.dataLakeStack.goldBucket.bucketName,
            '--execution-id', sfn.JsonPath.stringAt('$$.Execution.Id')
          ],
          sparkSubmitParameters: [
            '--conf', 'spark.sql.adaptive.enabled=true',
            '--conf', 'spark.sql.adaptive.coalescePartitions.enabled=true',
            '--conf', 'spark.sql.adaptive.skewJoin.enabled=true',
            '--conf', 'spark.sql.parquet.compression=snappy',
            '--conf', 'spark.executor.cores=4',
            '--conf', 'spark.executor.memory=16g'
          ]
        }
      },
      configurationOverrides: {
        applicationConfiguration: [
          {
            classification: 'spark-defaults',
            properties: {
              'spark.driver.memory': '16g',
              'spark.driver.maxResultSize': '4g',
              'spark.sql.adaptive.enabled': 'true',
              'spark.sql.adaptive.coalescePartitions.enabled': 'true',
              'spark.sql.adaptive.skewJoin.enabled': 'true'
            }
          }
        ],
        monitoringConfiguration: {
          managedPersistenceMonitoringConfiguration: {
            s3MonitoringConfiguration: {
              logUri: `s3://${props.dataLakeStack.bronzeBucket.bucketName}/emr-logs/`
            }
          }
        }
      },
      integrationPattern: sfn.IntegrationPattern.RUN_JOB,
      resultPath: '$.dataProcessingResult'
    });

    const validation = new sfn.Choice(this, DefaultIdBuilder.build(`validate-${dataProcessingStageName}`), {
      stateName: `validate-${dataProcessingStageName}`
    });

    validation.when(sfn.Condition.stringEquals('$.dataProcessingResult.status', 'SUCCESS'), new sfn.Succeed(this, DefaultIdBuilder.build(`${dataProcessingStageName}-success`), {
      comment: 'Data processing completed successfully'
    }));

    validation.otherwise(new sfn.Fail(this, DefaultIdBuilder.build(`${dataProcessingStageName}-failed`), {
      error: 'DataProcessingFailed',
      cause: 'Data processing stage failed',
      comment: 'Data processing stage encountered an error'
    }));

    this.workflow = sfn.Chain
      .start(dataProcessingTask)
      .next(validation);
  }
}