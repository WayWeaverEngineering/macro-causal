import { Duration, RemovalPolicy, Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { ConstructIdBuilder } from '@wayweaver/ariadne';
import { MACRO_CAUSAL_CONSTANTS } from '../utils/Constants';

export interface DataLakeStackProps extends StackProps {
  accountId: string;
  region: string;
  idBuilder: ConstructIdBuilder;
}

export class DataLakeStack extends Stack {
  public readonly bronzeBucket: s3.Bucket;
  public readonly silverBucket: s3.Bucket;
  public readonly goldBucket: s3.Bucket;
  public readonly artifactsBucket: s3.Bucket;

  constructor(scope: Construct, id: string, props: DataLakeStackProps) {
    super(scope, id, props);

    // Bronze bucket for raw data
    const bronzeBucketId = props.idBuilder.build('bronze-bucket');
    this.bronzeBucket = new s3.Bucket(this, bronzeBucketId, {
      bucketName: `${bronzeBucketId}-${props.accountId}`,
      versioned: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      lifecycleRules: [
        {
          id: 'DataRetention',
          enabled: true,
          noncurrentVersionExpiration: Duration.days(MACRO_CAUSAL_CONSTANTS.S3.DATA_RETENTION_DAYS),
          transitions: [
            {
              storageClass: s3.StorageClass.INFREQUENT_ACCESS,
              transitionAfter: Duration.days(30)
            },
            {
              storageClass: s3.StorageClass.GLACIER,
              transitionAfter: Duration.days(60)
            }
          ]
        }
      ],
      removalPolicy: RemovalPolicy.RETAIN,
      autoDeleteObjects: false
    });

    // Silver bucket for processed data
    const silverBucketId = props.idBuilder.build('silver-bucket');
    this.silverBucket = new s3.Bucket(this, silverBucketId, {
      bucketName: `${silverBucketId}-${props.accountId}`,
      versioned: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      lifecycleRules: [
        {
          id: 'DataRetention',
          enabled: true,
          noncurrentVersionExpiration: Duration.days(MACRO_CAUSAL_CONSTANTS.S3.DATA_RETENTION_DAYS),
          transitions: [
            {
              storageClass: s3.StorageClass.INFREQUENT_ACCESS,
              transitionAfter: Duration.days(30)
            }
          ]
        }
      ],
      removalPolicy: RemovalPolicy.RETAIN,
      autoDeleteObjects: false
    });

    // Gold bucket for feature-engineered data
    const goldBucketId = props.idBuilder.build('gold-bucket');
    this.goldBucket = new s3.Bucket(this, goldBucketId, {
      bucketName: `${goldBucketId}-${props.accountId}`,
      versioned: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      lifecycleRules: [
        {
          id: 'DataRetention',
          enabled: true,
          noncurrentVersionExpiration: Duration.days(MACRO_CAUSAL_CONSTANTS.S3.DATA_RETENTION_DAYS)
        }
      ],
      removalPolicy: RemovalPolicy.RETAIN,
      autoDeleteObjects: false
    });

    // Artifacts bucket for model storage
    const artifactsBucketId = props.idBuilder.build('artifacts-bucket');
    this.artifactsBucket = new s3.Bucket(this, artifactsBucketId, {
      bucketName: `${artifactsBucketId}-${props.accountId}`,
      versioned: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: RemovalPolicy.RETAIN,
      lifecycleRules: [
        {
          id: 'model-retention',
          enabled: true,
          noncurrentVersionExpiration: Duration.days(MACRO_CAUSAL_CONSTANTS.S3.MODEL_RETENTION_DAYS),
          noncurrentVersionTransitions: [
            {
              storageClass: s3.StorageClass.INFREQUENT_ACCESS,
              transitionAfter: Duration.days(30),
            },
          ],
        },
      ],
    });
  }
}
