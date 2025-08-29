import { Stage, StageProps } from "aws-cdk-lib";
import { Construct } from "constructs";
import { DefaultIdBuilder } from "../../utils/Naming";
import {
  COMMON_UTILS_LAMBDA_LAYER_NAME,
  PrebuiltLambdaLayersStack,
  LayerArns
} from "@wayweaver/ariadne";
import { CloudFrontDistributionStack } from "../stacks/CloudFrontDistributionStack";
import { DataLakeStack } from "../stacks/DataLakeStack";
import { VPCStack } from "../stacks/VPCStack";
import { APIDataCollectionStack } from "../stacks/APIDataCollectionStack";
import { MLTrainingStack } from "../stacks/MLTrainingStack";
import { InferenceStack } from "../stacks/InferenceStack";
import { MonitoringStack } from "../stacks/MonitoringStack";

interface DeploymentStageProps extends StageProps {
  prebuiltLambdaLayerArns: LayerArns;
  environment: string;
}

export class DeploymentStage extends Stage {
  constructor(scope: Construct, id: string, props: DeploymentStageProps) {
    super(scope, id, props);

    // Create CloudFront Distribution stack to host the website's build artifacts
    //const cloudFrontStackId = DefaultIdBuilder.build('cloudfront-dist-stack');
    //new CloudFrontDistributionStack(this, cloudFrontStackId);

    const lambdaLayersStackId = DefaultIdBuilder.build('lambda-layers-stack')
    const lambdaLayersStack = new PrebuiltLambdaLayersStack(this, lambdaLayersStackId, {
      arns: props.prebuiltLambdaLayerArns
    })

    const commonUtilsLambdaLayer = lambdaLayersStack.getLayer(COMMON_UTILS_LAMBDA_LAYER_NAME)

    // VPC Stack (foundation)
    const vpcStack = new VPCStack(this, DefaultIdBuilder.build('vpc-stack'), {
      environment: props.environment,
      accountId: this.account || '',
      region: this.region || 'us-east-1',
      env: {
        account: this.account,
        region: this.region
      }
    });

    // Data Lake Stack
    const dataLakeStack = new DataLakeStack(this, DefaultIdBuilder.build('data-lake-stack'), {
      environment: props.environment,
      vpcStack: vpcStack,
      env: {
        account: this.account,
        region: this.region
      }
    });

    // API Data Collection Stack
    const apiDataCollectionStack = new APIDataCollectionStack(this, DefaultIdBuilder.build('api-data-collection-stack'), {
      environment: props.environment,
      vpcStack: vpcStack,
      bronzeBucket: dataLakeStack.dataLake.bronzeBucket,
      env: {
        account: this.account,
        region: this.region
      }
    });

    // ML Training Stack
    const mlTrainingStack = new MLTrainingStack(this, DefaultIdBuilder.build('ml-training-stack'), {
      environment: props.environment,
      dataLakeStack: dataLakeStack,
      vpcStack: vpcStack,
      env: {
        account: this.account,
        region: this.region
      }
    });

    // Inference Stack
    const inferenceStack = new InferenceStack(this, DefaultIdBuilder.build('inference-stack'), {
      environment: props.environment,
      mlTrainingStack: mlTrainingStack,
      env: {
        account: this.account,
        region: this.region
      }
    });

    // Monitoring Stack
    const monitoringStack = new MonitoringStack(this, DefaultIdBuilder.build('monitoring-stack'), {
      environment: props.environment,
      inferenceStack: inferenceStack,
      mlTrainingStack: mlTrainingStack,
      vpcStack: vpcStack,
      env: {
        account: this.account,
        region: this.region
      }
    });

    // Add dependencies
    dataLakeStack.addDependency(vpcStack);
    apiDataCollectionStack.addDependency(dataLakeStack);
    apiDataCollectionStack.addDependency(vpcStack);
    mlTrainingStack.addDependency(dataLakeStack);
    mlTrainingStack.addDependency(vpcStack);
    inferenceStack.addDependency(mlTrainingStack);
    monitoringStack.addDependency(inferenceStack);
    monitoringStack.addDependency(mlTrainingStack);
  }
}