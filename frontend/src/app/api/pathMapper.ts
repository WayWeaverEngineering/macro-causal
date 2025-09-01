// API Gateway paths for Macro Causal Analysis
const apiGatewayBaseUrl = process.env.VITE_API_BASE_URL || "http://localhost:8000";

export const ApiGatewayPaths = {
  apiGatewayBaseUrl,
  causalAnalysisPath: '/causal-analysis',
  getCausalAnalysisStatusPath: (executionId: string) => `/causal-analysis/${executionId}`,
  regimeAnalysisPath: '/regime-analysis',
  uncertaintyAnalysisPath: '/uncertainty-analysis',
  macroVariablesPath: '/macro-variables',
  assetReturnsPath: '/asset-returns'
};
