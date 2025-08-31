import { Construct } from "constructs";
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ecrAssets from 'aws-cdk-lib/aws-ecr-assets';
import { DefaultIdBuilder } from '../../utils/Naming';

export interface EcsStageProps {
  stageName: string;
  cpu?: number;
  memoryLimitMiB?: number;
  portMappings?: { containerPort: number }[];
  environment?: { [key: string]: string };
}

export class EcsStageConstruct extends Construct {

  public readonly service: ecs.FargateService;

  constructor(scope: Construct, id: string, props: EcsStageProps) {
    super(scope, id);

    const clusterId = DefaultIdBuilder.build(`${props.stageName}-cluster`);
    const cluster = new ecs.Cluster(this, clusterId);

    const imgId = DefaultIdBuilder.build(`${props.stageName}-img`);
    const img = new ecrAssets.DockerImageAsset(this, imgId, {
      // IMPORTANT: the image path is relative to cdk.out
      directory: `../pipeline/${props.stageName}`,
      platform: ecrAssets.Platform.LINUX_AMD64,
    });

    const taskId = DefaultIdBuilder.build(`${props.stageName}-task`);
    const task = new ecs.FargateTaskDefinition(this, taskId, {
      cpu: props.cpu || 1024, memoryLimitMiB: props.memoryLimitMiB || 2048,
    });

    const containerId = DefaultIdBuilder.build(`${props.stageName}-container`);
    const container = task.addContainer(containerId, {
      image: ecs.ContainerImage.fromDockerImageAsset(img),
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: props.stageName }),
      portMappings: props.portMappings || [{ containerPort: 8080 }],
    });

    // Add environment variables if provided
    if (props.environment) {
      Object.entries(props.environment).forEach(([key, value]) => {
        container.addEnvironment(key, value);
      });
    }

    const serviceId = DefaultIdBuilder.build(`${props.stageName}-service`);
    this.service = new ecs.FargateService(this, serviceId, {
      cluster, taskDefinition: task, desiredCount: 0, // Don't run continuously - Step Functions will trigger tasks
      assignPublicIp: false,
    });
  }
}