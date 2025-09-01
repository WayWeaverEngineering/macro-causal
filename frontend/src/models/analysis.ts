// Macro Causal Analysis Models

export interface ExecutionStep {
  stepId: string;
  stepName: string;
  description: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  startTime?: Date;
  endTime?: Date;
  metadata?: Record<string, any>;
  error?: string;
  progress?: number;
}

export interface CausalAnalysisResult {
  summary: string;
  keyInsights: string[];
  causalEffect: CausalEffect;
  regimeAnalysis: RegimeAnalysis;
  uncertainty: UncertaintyEstimate;
  methodology: string;
  limitations: string[];
}

export interface CausalEffect {
  effect: number;
  confidenceInterval: [number, number];
  pValue: number;
  significance: 'high' | 'medium' | 'low';
  economicSignificance: 'high' | 'medium' | 'low';
  direction: 'positive' | 'negative' | 'neutral';
}

export interface RegimeAnalysis {
  currentRegime: number;
  regimeProbabilities: number[];
  regimeNames: string[];
  regimeEffects: Record<number, number>;
  regimeFeatures: Record<string, number>;
}

export interface UncertaintyEstimate {
  uncertainty: number;
  confidence: number;
  reliability: 'high' | 'medium' | 'low';
  factors: string[];
}

export interface MacroVariable {
  name: string;
  value: number;
  change: number;
  unit: string;
  source: string;
  frequency: string;
}

export interface AssetReturn {
  asset: string;
  return: number;
  volatility: number;
  correlation: number;
}

export interface AnalysisMetadata {
  executionTime: number;
  dataPointsAnalyzed: number;
  timePeriod: string;
  confidence: number;
  modelVersion: string;
}

// Backend Response Types
export interface BackendAnalysisResponse {
  success: boolean;
  message: string;
  query?: string;
  executionId?: string;
  status?: 'pending' | 'running' | 'completed' | 'failed';
  userQuery?: string;
  sessionId?: string;
  steps?: ExecutionStep[];
  currentStep?: ExecutionStep | null;
  result?: any;
  error?: string;
  createdAt?: string;
  updatedAt?: string;
  analysis?: CausalAnalysisResult;
  outOfScopeReason?: string;
  metadata?: {
    steps: ExecutionStep[];
    executionTime: number;
    dataPointsAnalyzed: number;
    timePeriod: string;
    confidence: number;
    modelVersion: string;
  };
  macroVariables?: MacroVariable[];
  assetReturns?: AssetReturn[];
}
