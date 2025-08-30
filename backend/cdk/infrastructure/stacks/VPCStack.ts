import { Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { VPCConstruct } from '../constructs/VPCConstruct';
import { DefaultIdBuilder } from '../../utils/Naming';

export interface VPCStackProps extends StackProps {
  accountId: string;
  region: string;
}

export class VPCStack extends Stack {
  public readonly vpcConstruct: VPCConstruct;

  constructor(scope: Construct, id: string, props: VPCStackProps) {
    super(scope, id, props);

    // Create VPC construct
    const vpcId = DefaultIdBuilder.build('vpc');
    this.vpcConstruct = new VPCConstruct(this, vpcId, {
      accountId: props.accountId,
      region: props.region,
      maxAzs: 2,
      natGateways: 1
    });
  }
}
