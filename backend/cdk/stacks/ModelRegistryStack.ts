import { Construct } from 'constructs';
import { Stack, StackProps, RemovalPolicy } from 'aws-cdk-lib';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import { DefaultIdBuilder } from '../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS } from '../utils/Constants';

export interface ModelRegistryStackProps extends StackProps {
  accountId: string;
  region: string;
}

export class ModelRegistryStack extends Stack {
  public readonly modelRegistryTable: dynamodb.Table;

  constructor(scope: Construct, id: string, props: ModelRegistryStackProps) {
    super(scope, id, props);

    // Create DynamoDB table for model registry
    const modelRegistryTableId = DefaultIdBuilder.build('model-registry-table');
    this.modelRegistryTable = new dynamodb.Table(this, modelRegistryTableId, {
      tableName: `${modelRegistryTableId}-${props.accountId}`,
      partitionKey: {
        name: 'model_id',
        type: dynamodb.AttributeType.STRING,
      },
      sortKey: {
        name: 'created_at',
        type: dynamodb.AttributeType.STRING,
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.RETAIN,
      timeToLiveAttribute: MACRO_CAUSAL_CONSTANTS.DYNAMODB.TTL_ATTRIBUTE,
    });

    // Add GSI for querying by execution_id
    this.modelRegistryTable.addGlobalSecondaryIndex({
      indexName: 'ExecutionIdIndex',
      partitionKey: {
        name: 'execution_id',
        type: dynamodb.AttributeType.STRING,
      },
      sortKey: {
        name: 'created_at',
        type: dynamodb.AttributeType.STRING,
      },
      projectionType: dynamodb.ProjectionType.ALL,
    });

    // Add GSI for querying by status
    this.modelRegistryTable.addGlobalSecondaryIndex({
      indexName: 'StatusIndex',
      partitionKey: {
        name: 'status',
        type: dynamodb.AttributeType.STRING,
      },
      sortKey: {
        name: 'created_at',
        type: dynamodb.AttributeType.STRING,
      },
      projectionType: dynamodb.ProjectionType.ALL,
    });
  }
}
