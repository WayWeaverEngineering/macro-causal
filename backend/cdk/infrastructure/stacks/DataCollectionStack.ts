import { Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { DataCollectionConstruct } from '../constructs/DataCollectionConstruct';
import { VPCStack } from './VPCStack';
import { DefaultIdBuilder } from '../../utils/Naming';

export interface DataCollectionStackProps extends StackProps {
  vpcStack: VPCStack;
  bronzeBucket: any; // s3.Bucket
}

export class DataCollectionStack extends Stack {
  public readonly dataCollection: DataCollectionConstruct;

  constructor(scope: Construct, id: string, props: DataCollectionStackProps) {
    super(scope, id, props);

    // API Data Collection construct
    const dataCollectionConstructId = DefaultIdBuilder.build('data-collection');
    this.dataCollection = new DataCollectionConstruct(this, dataCollectionConstructId, {
      bronzeBucket: props.bronzeBucket,
      vpc: props.vpcStack.vpcConstruct.vpc,
      securityGroup: props.vpcStack.vpcConstruct.securityGroup
    });
  }
}
