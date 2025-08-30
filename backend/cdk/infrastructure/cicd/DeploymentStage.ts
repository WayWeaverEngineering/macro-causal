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
import { MLTrainingStack } from "../stacks/MLTrainingStack";
import { InferenceStack } from "../stacks/InferenceStack";
import { MonitoringStack } from "../stacks/MonitoringStack";
import { DataCollectionStack } from "../stacks/DataCollectionStack";

interface DeploymentStageProps extends StageProps {
  prebuiltLambdaLayerArns: LayerArns;
}

export class DeploymentStage extends Stage {
  constructor(scope: Construct, id: string, props: DeploymentStageProps) {
    super(scope, id, props);

    // Create CloudFront Distribution stack to host the website's build artifacts
    const cloudFrontStackId = DefaultIdBuilder.build('cloudfront-dist-stack');
    new CloudFrontDistributionStack(this, cloudFrontStackId);

    const lambdaLayersStackId = DefaultIdBuilder.build('lambda-layers-stack')
    const lambdaLayersStack = new PrebuiltLambdaLayersStack(this, lambdaLayersStackId, {
      arns: props.prebuiltLambdaLayerArns
    })

    const commonUtilsLambdaLayer = lambdaLayersStack.getLayer(COMMON_UTILS_LAMBDA_LAYER_NAME)

    // Data Lake Stack
    const dataLakeStack = new DataLakeStack(this, DefaultIdBuilder.build('data-lake-stack'), {
      env: {
        account: this.account,
        region: this.region
      }
    });

    // API Data Collection Stack
    const dataCollectionStack = new DataCollectionStack(this, DefaultIdBuilder.build('data-collection-stack'), {
      bronzeBucket: dataLakeStack.dataLake.bronzeBucket
    });

    /*
    // ML Training Stack
    const mlTrainingStack = new MLTrainingStack(this, DefaultIdBuilder.build('ml-training-stack'), {
      dataLakeStack: dataLakeStack,
      vpcStack: vpcStack,
      env: {
        account: this.account,
        region: this.region
      }
    });

    // Inference Stack
    const inferenceStack = new InferenceStack(this, DefaultIdBuilder.build('inference-stack'), {
      mlTrainingStack: mlTrainingStack,
      vpcStack: vpcStack,
      env: {
        account: this.account,
        region: this.region
      }
    });

    // Monitoring Stack
    const monitoringStack = new MonitoringStack(this, DefaultIdBuilder.build('monitoring-stack'), {
      inferenceStack: inferenceStack,
      mlTrainingStack: mlTrainingStack,
      vpcStack: vpcStack,
      env: {
        account: this.account,
        region: this.region
      }
    });
    */

    // Add dependencies
    dataCollectionStack.addDependency(dataLakeStack);
    //mlTrainingStack.addDependency(dataLakeStack);
    //mlTrainingStack.addDependency(vpcStack);
    //inferenceStack.addDependency(mlTrainingStack);
    //monitoringStack.addDependency(inferenceStack);
    //monitoringStack.addDependency(mlTrainingStack);
  }
}