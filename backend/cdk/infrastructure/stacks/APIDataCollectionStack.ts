import { Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { RESOURCE_NAMES } from '../../utils/Constants';
import { APIDataCollectionConstruct } from '../constructs/APIDataCollectionConstruct';
import { VPCStack } from './VPCStack';

export interface APIDataCollectionStackProps extends StackProps {
  environment: string;
  vpcStack?: VPCStack;
  bronzeBucket: any; // s3.Bucket
}

export class APIDataCollectionStack extends Stack {
  public readonly apiDataCollection: APIDataCollectionConstruct;

  constructor(scope: Construct, id: string, props: APIDataCollectionStackProps) {
    super(scope, id, props);

    // API Data Collection construct
    this.apiDataCollection = new APIDataCollectionConstruct(this, RESOURCE_NAMES.API_DATA_COLLECTION_CONSTRUCT, {
      environment: props.environment,
      bronzeBucket: props.bronzeBucket,
      vpc: props.vpcStack?.vpcConstruct.vpc,
      securityGroup: props.vpcStack?.vpcConstruct.securityGroup
    });

    // Add tags to the stack
    this.tags.setTag('Environment', props.environment);
    this.tags.setTag('Project', 'MacroCausal');
    this.tags.setTag('Component', 'APIDataCollection');
  }
}
