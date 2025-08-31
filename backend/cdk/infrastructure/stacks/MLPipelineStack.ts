import { Construct } from 'constructs';
import { Stack, StackProps } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../../utils/Naming';
import { PipelineStageConstruct } from '../constructs/PipelineStageConstruct';
import { DataLakeStack } from './DataLakeStack';
import { AwsConfig } from '../configs/AwsConfig';
import { Effect, PolicyStatement } from 'aws-cdk-lib/aws-iam';

export interface MLPipelineStackProps extends StackProps {
  dataLakeStack: DataLakeStack;
}

export class MLPipelineStack extends Stack {

  constructor(scope: Construct, id: string, props: MLPipelineStackProps) {
    super(scope, id, props);

    const dataCollectionStage = new PipelineStageConstruct(
      this, DefaultIdBuilder.build('data-collection-stage'), {
      stageName: 'data-collection',
      environment: {
        BRONZE_BUCKET: props.dataLakeStack.bronzeBucket.bucketName,
        API_SECRETS_ARN: AwsConfig.FRED_API_SECRET_ARN
      }
    });

    // IAM policy statement to allow pipeline stages to fetch secrets from Secret Manager
    const secretAccessStatement = new PolicyStatement({
      effect: Effect.ALLOW,
      actions: ['secretsmanager:GetSecretValue'],
      resources: [
        AwsConfig.FRED_API_SECRET_ARN
      ]
    })

    // Add secret access statement to data collection stage task role
    dataCollectionStage.service.taskDefinition.taskRole.addToPrincipalPolicy(secretAccessStatement);

    // Enable data collection stage to write raw data to bronze bucket
    props.dataLakeStack.bronzeBucket.grantReadWrite(dataCollectionStage.service.taskDefinition.taskRole);
  }
}
