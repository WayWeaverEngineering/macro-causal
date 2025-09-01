

export interface ExecutionStep {
  stepId: string;
  stepName: string;
  description: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  startTime?: Date;
  endTime?: Date;
  metadata?: Record<string, any>;
  error?: string;
}

export interface PromptAnalysis {
  isInScope: boolean;
  reasoning: string;
  analysisType: 'treatment_effect' | 'regime_classification' | 'uncertainty_estimation' | 'other';
  complexity: 'simple' | 'moderate' | 'complex';
  requiredData: string[];
  estimatedSteps: number;
}

export interface ModelInputs {
  xLearner?: XLearnerInputs;
  regimeClassifier?: RegimeClassifierInputs;
  uncertaintyEstimator?: UncertaintyEstimatorInputs;
}

export interface XLearnerInputs {
  treatment_variables: string[];
  outcome_variables: string[];
  confounders: string[];
  time_periods: {
    start: string;
    end: string;
  };
}

export interface RegimeClassifierInputs {
  market_indicators: string[];
  lookback_periods: number;
  regime_count: number;
}

export interface UncertaintyEstimatorInputs {
  base_estimates: string[];
  confidence_level: number;
  bootstrap_samples: number;
}

export interface ModelResults {
  xLearner?: XLearnerResults;
  regimeClassifier?: RegimeClassifierResults;
  uncertaintyEstimator?: UncertaintyEstimatorResults;
}

export interface XLearnerResults {
  treatment_effect: number;
  confidence_interval: [number, number];
  p_value: number;
  standard_error: number;
}

export interface RegimeClassifierResults {
  regime_probabilities: number[];
  predicted_regime: number;
  confidence: number;
  regime_characteristics: Record<string, any>;
}

export interface UncertaintyEstimatorResults {
  uncertainty_estimate: number;
  confidence_interval: [number, number];
  reliability_score: number;
}

export interface AgentState {
  userQuery: string;
  executionSteps: ExecutionStep[];
  currentStep: ExecutionStep | null;
  metadata: Record<string, any>;
  isInScope: boolean;
  promptAnalysis: PromptAnalysis | null;
  generatedInputs: ModelInputs | null;
  modelResults: ModelResults | null;
  finalResponse: string | null;
  error: string | null;
}

export interface QueryResponse {
  success: boolean;
  response?: string;
  modelResults?: ModelResults;
  metadata?: {
    query: string;
    analysisType: string;
    complexity: string;
    executionTime: number;
    stepsCompleted: number;
    totalSteps: number;
    timestamp: string;
  };
  error?: string;
  outOfScopeReason?: string;
}

export interface ScopeCheckResult {
  inScope: boolean;
  reason: string;
  suggestedAlternatives?: string[];
}
