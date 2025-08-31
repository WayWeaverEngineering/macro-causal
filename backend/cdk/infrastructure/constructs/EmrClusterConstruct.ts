import { Construct } from 'constructs';
import * as emrserverless from 'aws-cdk-lib/aws-emrserverless';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import { DefaultIdBuilder } from '../../utils/Naming';

export interface EmrClusterProps {
  name: string;
  imageUri: string;
  bronzeBucket: s3.Bucket;
  silverBucket: s3.Bucket;
  goldBucket: s3.Bucket;
}

export class EmrClusterConstruct extends Construct {
  public readonly application: emrserverless.CfnApplication;
  public readonly executionRole: iam.Role;

  constructor(scope: Construct, id: string, props: EmrClusterProps) {
    super(scope, id);

    // Create IAM role for EMR Serverless execution
    this.executionRole = new iam.Role(this, DefaultIdBuilder.build('emr-serverless-execution-role'), {
      assumedBy: new iam.ServicePrincipal('emr-serverless.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonEMRServerlessServiceRolePolicy')
      ]
    });

    // Grant S3 access to EMR Serverless
    props.bronzeBucket.grantRead(this.executionRole);
    props.silverBucket.grantReadWrite(this.executionRole);
    props.goldBucket.grantReadWrite(this.executionRole);

    // Add CloudWatch Logs permissions
    this.executionRole.addToPolicy(new iam.PolicyStatement({
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

    

    // Create EMR Serverless Application
    const emrServerlessApplicationId = DefaultIdBuilder.build('emr-serverless-app');
    this.application = new emrserverless.CfnApplication(this, emrServerlessApplicationId, {
      name: props.name,
      type: 'SPARK',
      releaseLabel: `${emrServerlessApplicationId}-emr-7.0.0`,
      initialCapacity: [
        {
          key: 'DRIVER',
          value: {
            workerCount: 1,
            workerConfiguration: {
              cpu: '4vCPU',
              memory: '16GB',
              disk: '64GB'
            }
          }
        },
        {
          key: 'EXECUTOR',
          value: {
            workerCount: 2,
            workerConfiguration: {
              cpu: '4vCPU',
              memory: '16GB',
              disk: '64GB'
            }
          }
        }
      ],
      maximumCapacity: {
        cpu: '200vCPU',
        memory: '800GB',
        disk: '3200GB'
      },
      autoStopConfiguration: {
        enabled: true,
        idleTimeoutMinutes: 5
      },
      autoStartConfiguration: {
        enabled: true
      },
      imageConfiguration: {
        imageUri: props.imageUri
      }
    });
  }
}