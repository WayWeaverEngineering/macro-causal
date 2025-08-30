import { Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { MonitoringConstruct } from '../constructs/MonitoringConstruct';
import { DefaultIdBuilder } from '../../utils/Naming';

export interface MonitoringStackProps extends StackProps {
  inferenceStack: any; // InferenceStack
  mlTrainingStack: any; // MLTrainingStack
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
    });
  }
}
