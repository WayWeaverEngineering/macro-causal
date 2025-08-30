import { Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { ApiDataCollectionConstruct } from '../constructs/ApiDataCollectionConstruct';
import { VPCStack } from './VPCStack';
import { DefaultIdBuilder } from '../../utils/Naming';

export interface ApiDataCollectionStackProps extends StackProps {
  vpcStack: VPCStack;
  bronzeBucket: any; // s3.Bucket
}

export class ApiDataCollectionStack extends Stack {
  public readonly apiDataCollection: ApiDataCollectionConstruct;

  constructor(scope: Construct, id: string, props: ApiDataCollectionStackProps) {
    super(scope, id, props);

    // API Data Collection construct
    const apiDataCollectionConstructId = DefaultIdBuilder.build('api-data-collection');
    this.apiDataCollection = new ApiDataCollectionConstruct(this, apiDataCollectionConstructId, {
      bronzeBucket: props.bronzeBucket,
      vpc: props.vpcStack.vpcConstruct.vpc,
      securityGroup: props.vpcStack.vpcConstruct.securityGroup
    });
  }
}
