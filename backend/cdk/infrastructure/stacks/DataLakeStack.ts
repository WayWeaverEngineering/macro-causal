import { Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { DataLakeConstruct } from '../constructs/DataLakeConstruct';
import { DataIngestionConstruct } from '../constructs/DataIngestionConstruct';
import { VPCStack } from './VPCStack';
import { DefaultIdBuilder } from '../../utils/Naming';

export interface DataLakeStackProps extends StackProps {
  vpcStack: VPCStack;
}

export class DataLakeStack extends Stack {
  public readonly dataLake: DataLakeConstruct;
  public readonly dataIngestion: DataIngestionConstruct;

  constructor(scope: Construct, id: string, props: DataLakeStackProps) {
    super(scope, id, props);

    // Data Lake construct
    const dataLakeId = DefaultIdBuilder.build('data-lake');
    this.dataLake = new DataLakeConstruct(this, dataLakeId, {
      accountId: this.account,
      region: this.region
    });

    // Data Ingestion construct
    const dataIngestionId = DefaultIdBuilder.build('data-ingestion');
    this.dataIngestion = new DataIngestionConstruct(this, dataIngestionId, {
      bronzeBucket: this.dataLake.bronzeBucket,
      emrApplication: this.dataLake.emrApplication,
      emrRole: this.dataLake.emrRole,
      vpc: props.vpcStack.vpcConstruct.vpc,
      securityGroup: props.vpcStack.vpcConstruct.securityGroup
    });
  }
}
