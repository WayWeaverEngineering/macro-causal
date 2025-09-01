export interface QueryRequest {
  query: string;
  sessionId?: string;
  options?: QueryOptions;
}

export interface QueryOptions {
  confidenceLevel?: number;
  includeRawResults?: boolean;
  analysisDepth?: 'basic' | 'detailed' | 'comprehensive';
}

export interface QueryMetadata {
  query: string;
  analysisType: string;
  complexity: string;
  executionTime: number;
  stepsCompleted: number;
  totalSteps: number;
  timestamp: string;
}

export interface AnalysisRequest {
  query: string;
  sessionId?: string;
  options?: QueryOptions;
}

export interface AnalysisResponse {
  success: boolean;
  executionId: string;
  message: string;
}
