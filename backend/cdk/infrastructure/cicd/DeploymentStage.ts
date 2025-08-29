import { Stage } from "aws-cdk-lib";
import { Construct } from "constructs";
import { DefaultIdBuilder } from "../../utils/Naming";
import { CloudFrontDistributionStack } from "../stacks/CloudFrontDistributionStack";

export class DeploymentStage extends Stage {
  constructor(scope: Construct, id: string) {
    super(scope, id);

    // Create CloudFront Distribution stack to host the website's build artifacts
    const cloudFrontStackId = DefaultIdBuilder.build('cloudfront-dist-stack');
    new CloudFrontDistributionStack(this, cloudFrontStackId);
  }
}