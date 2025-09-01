// API Gateway paths for Macro Causal Analysis
const apiGatewayBaseUrl = "https://macro-ai-analyst-api.wayweaver.com"

export const ApiGatewayPaths = {
  apiGatewayBaseUrl,
  causalAnalysisPath: '/causal-analysis',
  getCausalAnalysisStatusPath: (executionId: string) => `/causal-analysis/${executionId}`,
  regimeAnalysisPath: '/regime-analysis',
  uncertaintyAnalysisPath: '/uncertainty-analysis',
  macroVariablesPath: '/macro-variables',
  assetReturnsPath: '/asset-returns'
};
