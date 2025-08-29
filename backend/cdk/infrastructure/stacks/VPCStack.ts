import { Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { VPCConstruct } from '../constructs/VPCConstruct';
import { DefaultIdBuilder } from '../../utils/Naming';
import { MACRO_CAUSAL_CONSTANTS, RESOURCE_NAMES } from '../../utils/Constants';

export interface VPCStackProps extends StackProps {
  environment: string;
  accountId: string;
  region: string;
}

export class VPCStack extends Stack {
  public readonly vpcConstruct: VPCConstruct;

  constructor(scope: Construct, id: string, props: VPCStackProps) {
    super(scope, id, props);

    // Create VPC construct
    this.vpcConstruct = new VPCConstruct(this, RESOURCE_NAMES.VPC_CONSTRUCT, {
      environment: props.environment,
      accountId: props.accountId,
      region: props.region,
      maxAzs: 2,
      natGateways: 1
    });

    // Add tags to the stack
    this.tags.setTag('Environment', props.environment);
    this.tags.setTag('Project', 'MacroCausal');
    this.tags.setTag('Component', 'VPC');
  }
}
