import { Construct } from 'constructs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import { Duration, RemovalPolicy } from 'aws-cdk-lib';
import { DefaultIdBuilder } from '../../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS } from '../../utils/Constants';
import * as ec2 from 'aws-cdk-lib/aws-ec2';

export interface ModelSavingProps {
  accountId: string;
  region: string;
  trainingRole: iam.Role;
  vpc: ec2.IVpc;
  securityGroup: ec2.ISecurityGroup;
}

export class ModelSavingConstruct extends Construct {
  public readonly artifactsBucket: s3.Bucket;
  public readonly registryTable: dynamodb.Table;
  public readonly modelRegistrar: lambda.Function;
  public readonly modelPromoter: lambda.Function;

  constructor(scope: Construct, id: string, props: ModelSavingProps) {
    super(scope, id);

    // S3 bucket for model artifacts
    const artifactsBucketId = DefaultIdBuilder.build('artifacts-bucket');
    this.artifactsBucket = new s3.Bucket(this, artifactsBucketId, {
      bucketName: `${artifactsBucketId}-${props.accountId}`,
      versioned: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      lifecycleRules: [
        {
          id: 'ModelRetention',
          enabled: true,
          noncurrentVersionExpiration: Duration.days(MACRO_CAUSAL_CONSTANTS.S3.MODEL_RETENTION_DAYS),
          transitions: [
            {
              storageClass: s3.StorageClass.INFREQUENT_ACCESS,
              transitionAfter: Duration.days(30)
            },
            {
              storageClass: s3.StorageClass.GLACIER,
              transitionAfter: Duration.days(90)
            }
          ]
        }
      ],
      removalPolicy: RemovalPolicy.RETAIN,
      autoDeleteObjects: false
    });

    // DynamoDB table for model registry
    const registryTableId = DefaultIdBuilder.build('model-registry-table');
    this.registryTable = new dynamodb.Table(this, registryTableId, {
      tableName: registryTableId,
      partitionKey: { name: 'modelId', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'version', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      timeToLiveAttribute: MACRO_CAUSAL_CONSTANTS.DYNAMODB.TTL_ATTRIBUTE,
      removalPolicy: RemovalPolicy.RETAIN,
      pointInTimeRecovery: true
    });

    // Global Secondary Index for querying by status and date
    this.registryTable.addGlobalSecondaryIndex({
      indexName: 'StatusDateIndex',
      partitionKey: { name: 'status', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'createdAt', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.ALL
    });

    // Global Secondary Index for querying by model type
    this.registryTable.addGlobalSecondaryIndex({
      indexName: 'ModelTypeIndex',
      partitionKey: { name: 'modelType', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'createdAt', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.ALL
    });

    // Lambda function for model registration
    const modelRegistrarId = DefaultIdBuilder.build('model-registrar');
    this.modelRegistrar = new lambda.Function(this, modelRegistrarId, {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'index.handler',
      code: lambda.Code.fromAsset('../lambda/model-registry'),
      environment: {
        DYNAMODB_TABLE: this.registryTable.tableName,
        S3_BUCKET: this.artifactsBucket.bucketName,
      },
      timeout: Duration.minutes(MACRO_CAUSAL_CONSTANTS.LAMBDA.TIMEOUT_MINUTES),
      memorySize: MACRO_CAUSAL_CONSTANTS.LAMBDA.MEMORY_MB,
      vpc: props.vpc,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS
      },
      securityGroups: [props.securityGroup]
    });

    // Lambda function for model promotion
    const modelPromoterId = DefaultIdBuilder.build('model-promoter');
    this.modelPromoter = new lambda.Function(this, modelPromoterId, {
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'model_promoter.handler',
      code: lambda.Code.fromAsset('../lambda/model-registry'),
      environment: {
        DYNAMODB_TABLE: this.registryTable.tableName,
        S3_BUCKET: this.artifactsBucket.bucketName,
      },
      timeout: Duration.minutes(MACRO_CAUSAL_CONSTANTS.LAMBDA.TIMEOUT_MINUTES),
      memorySize: MACRO_CAUSAL_CONSTANTS.LAMBDA.MEMORY_MB,
      vpc: props.vpc,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS
      },
      securityGroups: [props.securityGroup]
    });

    // Grant permissions to Lambda functions
    this.registryTable.grantReadWriteData(this.modelRegistrar);
    this.registryTable.grantReadWriteData(this.modelPromoter);
    this.artifactsBucket.grantReadWrite(this.modelRegistrar);
    this.artifactsBucket.grantReadWrite(this.modelPromoter);

    // Grant permissions to training role
    this.artifactsBucket.grantReadWrite(props.trainingRole);
    this.registryTable.grantReadWriteData(props.trainingRole);

    // EventBridge rule for model registration (triggered by training completion)
    const modelRegistrationRuleId = DefaultIdBuilder.build('model-registration-rule');
    const modelRegistrationRule = new events.Rule(this, modelRegistrationRuleId, {
      ruleName: modelRegistrationRuleId,
      eventPattern: {
        source: ['aws.states'],
        detailType: ['Step Functions Execution Status Changed'],
        detail: {
          stateMachineArn: [{ suffix: 'training-workflow' }],
          status: ['SUCCEEDED']
        }
      },
      targets: [
        new targets.LambdaFunction(this.modelRegistrar, {
          event: events.RuleTargetInput.fromObject({
            modelId: events.EventField.fromPath('$.detail.name'),
            executionArn: events.EventField.fromPath('$.detail.executionArn'),
            status: events.EventField.fromPath('$.detail.status')
          })
        })
      ]
    });

    // EventBridge rule for model promotion (manual trigger)
    const modelPromotionRuleId = DefaultIdBuilder.build('model-promotion-rule');
    const modelPromotionRule = new events.Rule(this, modelPromotionRuleId, {
      ruleName: modelPromotionRuleId,
      eventPattern: {
        source: ['macro-causal.model-promotion'],
        detailType: ['ModelPromotionRequest']
      },
      targets: [
        new targets.LambdaFunction(this.modelPromoter, {
          event: events.RuleTargetInput.fromObject({
            modelId: events.EventField.fromPath('$.detail.modelId'),
            version: events.EventField.fromPath('$.detail.version'),
            targetStatus: events.EventField.fromPath('$.detail.targetStatus')
          })
        })
      ]
    });
  }
}
