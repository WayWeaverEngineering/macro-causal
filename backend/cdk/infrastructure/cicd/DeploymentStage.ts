import { Stage, StageProps } from "aws-cdk-lib";
import { Construct } from "constructs";
import { DefaultIdBuilder } from "../../utils/Naming";
import {
  COMMON_UTILS_LAMBDA_LAYER_NAME,
  PrebuiltLambdaLayersStack,
  LayerArns
} from "@wayweaver/ariadne";
import { CloudFrontDistributionStack } from "../stacks/CloudFrontDistributionStack";

interface DeploymentStageProps extends StageProps {
  prebuiltLambdaLayerArns: LayerArns
}

export class DeploymentStage extends Stage {
  constructor(scope: Construct, id: string, props: DeploymentStageProps) {
    super(scope, id,  props);

    // Create CloudFront Distribution stack to host the website's build artifacts
    const cloudFrontStackId = DefaultIdBuilder.build('cloudfront-dist-stack');
    new CloudFrontDistributionStack(this, cloudFrontStackId);

    const lambdaLayersStackId = DefaultIdBuilder.build('lambda-layers-stack')
    const lambdaLayersStack = new PrebuiltLambdaLayersStack(this, lambdaLayersStackId, {
      arns: props.prebuiltLambdaLayerArns
    })

    const commonUtilsLambdaLayer = lambdaLayersStack.getLayer(COMMON_UTILS_LAMBDA_LAYER_NAME)
  }
}