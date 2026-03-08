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
  CdkAppBuild,
  CloudFrontDistributionStack,
  RestApiStack,
  HARRY_FINANCE_ENVIRONMENT
} from '@wayweaver/ariadne';
import { Construct } from 'constructs';
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
    environment: options.environment,
    subDomain: 'macro-ai-analyst',
    prebuiltLambdaFunctions: prebuiltLambdaFunctions,
  });

  const dataLakeStackId = idBuilder.build('data-lake-stack');
  const dataLakeStack = new DataLakeStack(scope, dataLakeStackId, {
    idBuilder,
    accountId: options.environment.accountId,
    region: options.environment.region,
  });

  const modelRegistryStackId = idBuilder.build('model-registry-stack');
  const modelRegistryStack = new ModelRegistryStack(scope, modelRegistryStackId, {
    accountId: options.environment.accountId,
    region: options.environment.region,
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
    domainConfig: options.environment.domain,
    subDomain: 'macro-ai-analyst-api',
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
    preBuildCommands: [
      "echo Upgrading pip...",
      "python3 -m pip install --upgrade pip",
    ]
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

  await CdkAppBuild(appBuildOptions, [HARRY_FINANCE_ENVIRONMENT]);
}

main();