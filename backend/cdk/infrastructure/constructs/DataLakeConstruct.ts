import { Construct } from 'constructs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as emrserverless from 'aws-cdk-lib/aws-emrserverless';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Duration, RemovalPolicy } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS } from '../../utils/Constants';

export interface DataLakeProps {
  accountId: string;
  region: string;
}

export class DataLakeConstruct extends Construct {
  public readonly bronzeBucket: s3.Bucket;
  public readonly silverBucket: s3.Bucket;
  public readonly goldBucket: s3.Bucket;
  public readonly logsBucket: s3.Bucket;
  public readonly emrApplication: emrserverless.CfnApplication;
  public readonly emrRole: iam.Role;

  constructor(scope: Construct, id: string, props: DataLakeProps) {
    super(scope, id);

    // Bronze bucket for raw data
    const bronzeBucketId = DefaultIdBuilder.build('bronze-bucket');
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
    const silverBucketId = DefaultIdBuilder.build('silver-bucket');
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
    const goldBucketId = DefaultIdBuilder.build('gold-bucket');
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

    // Logs bucket for EMR and other logs
    const logsBucketId = DefaultIdBuilder.build('logs-bucket');
    this.logsBucket = new s3.Bucket(this, logsBucketId, {
      bucketName: `${logsBucketId}-${props.accountId}`,
      versioned: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      lifecycleRules: [
        {
          id: 'LogRetention',
          enabled: true,
          expiration: Duration.days(MACRO_CAUSAL_CONSTANTS.CLOUDWATCH.LOG_RETENTION_DAYS)
        }
      ],
      removalPolicy: RemovalPolicy.RETAIN,
      autoDeleteObjects: false
    });

    // EMR Serverless application
    /*
    const emrApplicationId = DefaultIdBuilder.build('emr-application');
    this.emrApplication = new emrserverless.CfnApplication(this, emrApplicationId, {
      type: 'SPARK',
      name: emrApplicationId,
      releaseLabel: MACRO_CAUSAL_CONSTANTS.EMR.RELEASE_LABEL,
      initialCapacity: [
        {
          key: 'DRIVER',
          value: {
            workerCount: 1,
            workerConfiguration: {
              cpu: MACRO_CAUSAL_CONSTANTS.EMR.DRIVER_CPU,
              memory: MACRO_CAUSAL_CONSTANTS.EMR.DRIVER_MEMORY
            }
          }
        }
      ],
      maximumCapacity: {
        cpu: MACRO_CAUSAL_CONSTANTS.EMR.MAX_CPU,
        memory: MACRO_CAUSAL_CONSTANTS.EMR.MAX_MEMORY
      },
      autoStopConfiguration: {
        enabled: true,
        idleTimeoutMinutes: 15
      },
      autoStartConfiguration: {
        enabled: true
      },
      imageConfiguration: {
        // More details on EMR on EKS releases here:
        // https://docs.aws.amazon.com/emr/latest/EMR-on-EKS-DevelopmentGuide/emr-eks-releases.html
        imageUri: `public.ecr.aws/emr-serverless/spark/emr-${MACRO_CAUSAL_CONSTANTS.EMR.RELEASE_LABEL}-latest`
      }
    });

    // EMR execution role
    const emrExecutionRoleId = DefaultIdBuilder.build('emr-execution-role');
    this.emrRole = new iam.Role(this, emrExecutionRoleId, {
      assumedBy: new iam.ServicePrincipal('emr-serverless.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonEMRServerlessServiceRolePolicy')
      ]
    });

    // Grant EMR role access to S3 buckets
    this.bronzeBucket.grantReadWrite(this.emrRole);
    this.silverBucket.grantReadWrite(this.emrRole);
    this.goldBucket.grantReadWrite(this.emrRole);
    this.logsBucket.grantReadWrite(this.emrRole);

    // Add bucket policies for EMR access
    [this.bronzeBucket, this.silverBucket, this.goldBucket, this.logsBucket].forEach(bucket => {
      bucket.addToResourcePolicy(new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        principals: [new iam.ServicePrincipal('emr-serverless.amazonaws.com')],
        actions: ['s3:GetObject', 's3:PutObject', 's3:DeleteObject'],
        resources: [bucket.arnForObjects('*')],
        conditions: {
          StringEquals: {
            'aws:SourceAccount': props.accountId
          }
        }
      }));
    });
    */
  }
}
