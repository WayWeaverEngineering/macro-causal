import { Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { RESOURCE_NAMES } from '../../utils/Constants';
import { DataLakeConstruct } from '../constructs/DataLakeConstruct';
import { DataIngestionConstruct } from '../constructs/DataIngestionConstruct';
import { VPCStack } from './VPCStack';

export interface DataLakeStackProps extends StackProps {
  environment: string;
  vpcStack?: VPCStack;
}

export class DataLakeStack extends Stack {
  public readonly dataLake: DataLakeConstruct;
  public readonly dataIngestion: DataIngestionConstruct;

  constructor(scope: Construct, id: string, props: DataLakeStackProps) {
    super(scope, id, props);

    // Data Lake construct
    this.dataLake = new DataLakeConstruct(this, RESOURCE_NAMES.DATA_LAKE_CONSTRUCT, {
      environment: props.environment,
      accountId: this.account,
      region: this.region
    });

    // Data Ingestion construct
    this.dataIngestion = new DataIngestionConstruct(this, RESOURCE_NAMES.DATA_INGESTION_CONSTRUCT, {
      environment: props.environment,
      bronzeBucket: this.dataLake.bronzeBucket,
      emrApplication: this.dataLake.emrApplication,
      emrRole: this.dataLake.emrRole,
      vpc: props.vpcStack?.vpcConstruct.vpc,
      securityGroup: props.vpcStack?.vpcConstruct.securityGroup
    });
  }
}
