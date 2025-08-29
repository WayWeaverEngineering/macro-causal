import { Duration, Stack, StackProps } from "aws-cdk-lib";
import { Construct } from "constructs";
import { BlockPublicAccess, Bucket } from "aws-cdk-lib/aws-s3";
import { join } from "path";
import { existsSync } from "fs";
import { BucketDeployment, Source } from "aws-cdk-lib/aws-s3-deployment";
import { AccessLevel, Distribution, OriginAccessIdentity, ViewerProtocolPolicy } from "aws-cdk-lib/aws-cloudfront";
import { S3BucketOrigin } from "aws-cdk-lib/aws-cloudfront-origins";
import { CanonicalUserPrincipal, PolicyStatement } from "aws-cdk-lib/aws-iam";
import { Certificate } from "aws-cdk-lib/aws-certificatemanager";
import { AwsConfig } from "../configs/AwsConfig";
import { DefaultIdBuilder } from "../../utils/Naming";

export class CloudFrontDistributionStack extends Stack {

  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const certificateId = DefaultIdBuilder.build('web-domain-certificate')
    const domainCertificate = Certificate.fromCertificateArn(
      this,
      certificateId,
      AwsConfig.WEB_DOMAIN_CERTIFICATE_ARN
    );

    const websiteDomainNames = [
      "www.sec-ai-analyst.wayweaver.com",
      "sec-ai-analyst.wayweaver.com"
    ]

    const distBucketId = DefaultIdBuilder.build('dist-bucket');
    const deploymentBucket = new Bucket(this, distBucketId, {
      bucketName: distBucketId,
      // Block direct public access
      blockPublicAccess: BlockPublicAccess.BLOCK_ALL,
    })

    // Grant access to the CloudFront distribution
    const s3BucketOaiId = DefaultIdBuilder.build('cloudfront-oai')
    const cloudFrontOAI = new OriginAccessIdentity(this, s3BucketOaiId, {
      comment: `Access Identity for '${distBucketId}'`,
    });

    // Attach the bucket policy to allow CloudFront OAI access
    deploymentBucket.addToResourcePolicy(
      new PolicyStatement({
        actions: ['s3:GetObject'],
        resources: [`${deploymentBucket.bucketArn}/*`],
        principals: [
          new CanonicalUserPrincipal(cloudFrontOAI.cloudFrontOriginAccessIdentityS3CanonicalUserId),
        ],
      })
    );

    const distributionId = DefaultIdBuilder.build('distribution');
    const distribution = new Distribution(this, distributionId, {
      defaultRootObject: 'index.html',
      domainNames: websiteDomainNames,
      certificate: domainCertificate,
      defaultBehavior: {
        origin: S3BucketOrigin.withOriginAccessControl(deploymentBucket, {
          originAccessLevels: [AccessLevel.READ]
        }),
        // Enforce HTTPS
        viewerProtocolPolicy: ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
      },
      // IMPORTANT: Because CloudFront is a CDN, it is only good for static hosting.
      // In static hosting, the subpaths of dynamic pages (such as React pages) can
      // only be accessed dynamically via links/buttons within the webpage. If the
      // user tries to access a subpath directly via a new browser tab, they will
      // into 403 (AccessDenied) errors. The workaround is to configure CloudFront
      // to redirect users from error pages back to the root page. More details here:
      // https://stackoverflow.com/questions/50299204/receive-accessdenied-when-trying-to-access-a-reload-or-refresh-or-one-in-new-tab/50302276#50302276
      errorResponses: [
        {
          httpStatus: 403, // Forbidden
          responseHttpStatus: 200,
          responsePagePath: '/index.html',
          ttl: Duration.seconds(0),
        },
        {
          httpStatus: 404, // Not Found
          responseHttpStatus: 200,
          responsePagePath: '/index.html',
          ttl: Duration.seconds(0),
        },
      ],
    });

    const uiDir = join(__dirname, '..', '..', '..', '..', 'frontend', 'dist');
    if (!existsSync(uiDir)) {
      throw new Error('UI assets directory not found: ' + uiDir)
    }

    const bucketDeploymentId = DefaultIdBuilder.build('dist-bucket-deployment');
    new BucketDeployment(this, bucketDeploymentId, {
      destinationBucket: deploymentBucket,
      sources: [ Source.asset(uiDir) ],
      distribution: distribution,
      // Invalidate all paths so that cached contents will be invalidated upon deployment.
      // Otherwise, CloudFront may attempt to serve stale contents that no longer exist,
      // leading to hard-to-debug errors such as 404 or 403.
      distributionPaths: ['/*']
    })
  }
}