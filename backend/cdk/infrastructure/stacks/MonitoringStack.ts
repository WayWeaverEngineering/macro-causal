import { Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { RESOURCE_NAMES } from '../../utils/Constants';
import { MonitoringConstruct } from '../constructs/MonitoringConstruct';

export interface MonitoringStackProps extends StackProps {
  environment: string;
  inferenceStack: any; // InferenceStack
  mlTrainingStack: any; // MLTrainingStack
}

export class MonitoringStack extends Stack {
  public readonly monitoring: MonitoringConstruct;

  constructor(scope: Construct, id: string, props: MonitoringStackProps) {
    super(scope, id, props);

    // Monitoring construct
    this.monitoring = new MonitoringConstruct(this, RESOURCE_NAMES.MONITORING_CONSTRUCT, {
      environment: props.environment,
      accountId: this.account,
      region: this.region,
      ecsCluster: props.inferenceStack.inference.ecsCluster,
      alb: props.inferenceStack.inference.alb,
      registryTable: props.mlTrainingStack.modelSaving.registryTable
    });
  }
}
