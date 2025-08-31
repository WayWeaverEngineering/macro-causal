import { CodePipeline, CodePipelineSource, ShellStep } from "aws-cdk-lib/pipelines";
import { Construct } from "constructs";
import { GitHubConfig } from "../configs/GithubConfig";
import { ComputeType, LinuxBuildImage } from "aws-cdk-lib/aws-codebuild";
import { PolicyStatement } from "aws-cdk-lib/aws-iam";
import { DefaultIdBuilder } from "../../utils/Naming";
import { DEFAULT_GITHUB_CODE_CONNECTION_ARN, DEFAULT_GITHUB_PRODUCTION_BRANCH } from "@wayweaver/ariadne";

export class CodeBuildPipeline extends CodePipeline {
  constructor(scope: Construct, id: string) {
    super(scope, id, {
      pipelineName: id,
      codeBuildDefaults: {
        buildEnvironment: {
          computeType: ComputeType.SMALL,
          // This build image uses Node 20+
          // which is needed once Node 18 reaches end of life
          buildImage: LinuxBuildImage.STANDARD_7_0,

        },
        rolePolicy: [
          // "codeartifact" permissions are needed to allow
          // CodeBuild to publish packages to CodeArtifact,
          // while "sts" permission is needed to allow
          // CodeBuild to obtain CodeArtifact credentials.
          // The "ssm:GetParameter" permission is to allow
          // the pipeline to obtain NPM package versions from SSM
          new PolicyStatement({
            actions: [
              'codeartifact:GetAuthorizationToken',
              'codeartifact:GetRepositoryEndpoint',
              'codeartifact:ReadFromRepository',
            ],
            resources: [
              'arn:aws:codeartifact:us-east-1:715067592333:domain/wayweaver-shared-artifacts',
              'arn:aws:codeartifact:us-east-1:715067592333:repository/wayweaver-shared-artifacts/npm-typescript',
            ],
          }),
          new PolicyStatement({
            actions: [
              'sts:GetServiceBearerToken'
            ],
            resources: [
              // Must allow CodeBuild to call sts:GetServiceBearerToken on its own identity
              'arn:aws:sts::715067592333:assumed-role/macro-causal-ci-cd-stack-macrocausalcodebuildpipeli*'
            ],
          }),
          new PolicyStatement({
            actions: [
              'ssm:GetParameter'
            ],
            resources: [
              'arn:aws:ssm:us-east-1:715067592333:parameter/npm-packages*',
              'arn:aws:ssm:us-east-1:715067592333:parameter/lambda-layers*',
              'arn:aws:ssm:us-east-1:715067592333:parameter/sqs-queues*'
            ],
          }),
        ],
      },
      // Enable CodeBuild to build the code using Docker
      dockerEnabledForSynth: true,
      synth: new ShellStep(DefaultIdBuilder.build('code-build-shell-script'), {
          input: CodePipelineSource.connection(
            GitHubConfig.GITHUB_REPO,
            DEFAULT_GITHUB_PRODUCTION_BRANCH,
            {
              connectionArn: DEFAULT_GITHUB_CODE_CONNECTION_ARN
            }
          ),
          commands: [
            // Print current directory for debugging
            "pwd",

            // Build the frontend
            "cd frontend",
            "npm ci",
            "npm run build",

            // Build the backend
            "cd ../backend/cdk",
            "echo Obtaining npm credentials...",
            "chmod +x ./scripts/npm-authenticate.sh",
            "./scripts/npm-authenticate.sh",
            "echo Synthesizing CDK stack...",
            "npm ci",
            "npm run build",
            "npx cdk synth"
          ],
          primaryOutputDirectory: "./backend/cdk/cdk.out"
      })
    })
  }
}