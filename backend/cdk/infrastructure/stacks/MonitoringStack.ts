import { Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { MonitoringConstruct } from '../constructs/MonitoringConstruct';
import { DefaultIdBuilder } from '../../utils/Naming';
import { VPCStack } from './VPCStack';

export interface MonitoringStackProps extends StackProps {
  inferenceStack: any; // InferenceStack
  mlTrainingStack: any; // MLTrainingStack
  vpcStack: VPCStack;
}

export class MonitoringStack extends Stack {
  public readonly monitoring: MonitoringConstruct;

  constructor(scope: Construct, id: string, props: MonitoringStackProps) {
    super(scope, id, props);

    // Monitoring construct
    const monitoringId = DefaultIdBuilder.build('monitoring');
    this.monitoring = new MonitoringConstruct(this, monitoringId, {
      accountId: this.account,
      region: this.region,
      ecsCluster: props.inferenceStack.inference.ecsCluster,
      alb: props.inferenceStack.inference.alb,
      registryTable: props.mlTrainingStack.modelSaving.registryTable,
      vpc: props.vpcStack.vpcConstruct.vpc,
      securityGroup: props.vpcStack.vpcConstruct.securityGroup
    });
  }
}
