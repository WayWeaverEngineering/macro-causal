import {
  requirePrebuiltResources,
  ConstructIdBuilder,
  DeploymentFunction,
  DeploymentOptions,
  createBuildConfigs,
  CdkAppBuildOptions,
  AWS_LAMBDA_LAYERS,
  INTEGRATION_LAMBDA_LAYERS,
  UTILS_LAMBDA_LAYERS,
  DEVOPS_ENVIRONMENT,
  CdkAppBuild,
  CloudFrontDistributionStack,
  RestApiStack
} from '@wayweaver/ariadne';
import { Construct } from 'constructs';
import { AwsConfig } from '../configs/AwsConfig';
import { DataLakeStack } from '../stacks/DataLakeStack';
import { ModelRegistryStack } from '../stacks/ModelRegistryStack';
import { MLPipelineStack } from '../stacks/MLPipelineStack';
import { AnalysisStack } from '../stacks/AnalysisStack';
import { LambdaIntegration } from 'aws-cdk-lib/aws-apigateway';

const deploy: DeploymentFunction = (
  scope: Construct,
  idBuilder: ConstructIdBuilder,
  options: DeploymentOptions
) => {
  const { prebuiltResources } = options;

  if (!prebuiltResources) {
    throw new Error('Prebuilt resources are required');
  }

  const {
    prebuiltLambdaLayers,
    prebuiltLambdaFunctions,
  } = requirePrebuiltResources(prebuiltResources, [
    'prebuiltLambdaLayers',
    'prebuiltLambdaFunctions',
  ] as const);

  // Create CloudFront Distribution stack to host the website's build artifacts
  const cloudFrontStackId = idBuilder.build('cloudfront-dist-stack');
  new CloudFrontDistributionStack(scope, cloudFrontStackId, {
    idBuilder,
    domainCertificateArn: AwsConfig.WEB_DOMAIN_CERTIFICATE_ARN,
    domainNames: [
      "www.macro-ai-analyst.harryfinance.ai",
      "macro-ai-analyst.harryfinance.ai"
    ],
    prebuiltLambdaFunctions: prebuiltLambdaFunctions,
  });

  const dataLakeStackId = idBuilder.build('data-lake-stack');
  const dataLakeStack = new DataLakeStack(scope, dataLakeStackId, {
    idBuilder,
    accountId: DEVOPS_ENVIRONMENT.accountId,
    region: DEVOPS_ENVIRONMENT.region,
  });

  const modelRegistryStackId = idBuilder.build('model-registry-stack');
  const modelRegistryStack = new ModelRegistryStack(scope, modelRegistryStackId, {
    accountId: DEVOPS_ENVIRONMENT.accountId,
    region: DEVOPS_ENVIRONMENT.region,
    idBuilder,
  });

  const mlPipelineStackId = idBuilder.build('ml-pipeline-stack');
  const mlPipelineStack = new MLPipelineStack(scope, mlPipelineStackId, {
    idBuilder,
    dataLakeStack,
    prebuiltLambdaLayers: prebuiltLambdaLayers,
    modelRegistryTable: modelRegistryStack.modelRegistryTable
  });

  // Create Analysis stack
  const analysisStackId = idBuilder.build('analysis-stack');
  const analysisStack = new AnalysisStack(scope, analysisStackId, {
    idBuilder,
    prebuiltLambdaLayers: prebuiltLambdaLayers,
  });

  const backendApiStackId = idBuilder.build('backend-api-stack');
  const backendApiStack = new RestApiStack(scope, backendApiStackId, {
    idBuilder,
    domainCertificateArn: AwsConfig.API_DOMAIN_CERTIFICATE_ARN,
    domainName: 'macro-ai-analyst-api.harryfinance.ai',
    apiDescription: 'REST API to expose Macro Causal backend functionalities',
    prebuiltLambdaFunctions: prebuiltLambdaFunctions,
  });

  const analysisResource = backendApiStack.restApi.root.resourceForPath('/analysis');
  analysisResource.addMethod('POST', new LambdaIntegration(analysisStack.analysisSchedulingLambda));
  const executionIdResource = analysisResource.addResource('{executionId}');
  executionIdResource.addMethod('GET', new LambdaIntegration(analysisStack.analysisStatusLambda));

  mlPipelineStack.addDependency(dataLakeStack);
  mlPipelineStack.addDependency(modelRegistryStack);
  backendApiStack.addDependency(analysisStack);
}

async function main() {
  const appName = "macro-causal"
  const gitHubRepo = "WayWeaverEngineering/macro-causal"

  const buildConfigs = createBuildConfigs(appName, gitHubRepo, {
    hasFrontendBuild: true,
  });

  const appBuildOptions: CdkAppBuildOptions = {
    deploy,
    buildConfigs,
    hasCloudflareDnsSync: true,
    prebuiltLambdaLayerNames: [
      INTEGRATION_LAMBDA_LAYERS.LANGCHAIN_LANGGRAPH_LAMBDA_LAYER,
      AWS_LAMBDA_LAYERS.AWS_ECS_LAMBDA_LAYER,
      AWS_LAMBDA_LAYERS.AWS_EMR_SERVERLESS_LAMBDA_LAYER,
      AWS_LAMBDA_LAYERS.AWS_DYNAMODB_LAMBDA_LAYER,
      AWS_LAMBDA_LAYERS.AWS_QUEUE_LAMBDA_LAYER,
      AWS_LAMBDA_LAYERS.AWS_OPENSEARCH_LAMBDA_LAYER,
      UTILS_LAMBDA_LAYERS.LAMBDA_UTILS_LAMBDA_LAYER,
    ]
  }

  await CdkAppBuild(appBuildOptions, [DEVOPS_ENVIRONMENT]);
}

main();