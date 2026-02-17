// API Gateway paths for Macro Causal Analysis
const apiGatewayBaseUrl = "https://macro-ai-analyst-api.harryfinance.ai"

export const ApiGatewayPaths = {
  apiGatewayBaseUrl,
  analysisPath: '/analysis',
  getAnalysisStatusPath: (executionId: string) => `/analysis/${executionId}`,
  // Legacy paths for backward compatibility
  causalAnalysisPath: '/analysis',
  getCausalAnalysisStatusPath: (executionId: string) => `/analysis/${executionId}`,
};
