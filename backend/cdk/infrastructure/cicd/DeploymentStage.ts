import { Stage, StageProps } from "aws-cdk-lib";
import { Construct } from "constructs";
import { DefaultIdBuilder } from "../../utils/Naming";
import {
  LayerArns,
  DEFAULT_REGION,
  AWS_ADMIN_ACCOUNT_ID,
  PrebuiltLambdaLayersStack,
} from "@wayweaver/ariadne";
import { CloudFrontDistributionStack } from "../stacks/CloudFrontDistributionStack";
import { DataLakeStack } from "../stacks/DataLakeStack";
import { MLPipelineStack } from "../stacks/MLPipelineStack";

interface DeploymentStageProps extends StageProps {
  prebuiltLambdaLayerArns: LayerArns;
}

export class DeploymentStage extends Stage {
  constructor(scope: Construct, id: string, props: DeploymentStageProps) {
    super(scope, id, props);

    // Create CloudFront Distribution stack to host the website's build artifacts
    const cloudFrontStackId = DefaultIdBuilder.build('cloudfront-dist-stack');
    new CloudFrontDistributionStack(this, cloudFrontStackId);

    const dataLakeStackId = DefaultIdBuilder.build('data-lake-stack');
    const dataLakeStack = new DataLakeStack(this, dataLakeStackId, {
      accountId: this.account ?? AWS_ADMIN_ACCOUNT_ID,
      region: this.region ?? DEFAULT_REGION,
    });

    const lambdaLayersStackId = DefaultIdBuilder.build('lambda-layers-stack')
    const lambdaLayersStack = new PrebuiltLambdaLayersStack(this, lambdaLayersStackId, {
      arns: props.prebuiltLambdaLayerArns
    })

    const mlPipelineStackId = DefaultIdBuilder.build('ml-pipeline-stack');
    const mlPipelineStack = new MLPipelineStack(this, mlPipelineStackId, {
      dataLakeStack,
      lambdaLayersStack
    });

    mlPipelineStack.addDependency(dataLakeStack);
  }
}