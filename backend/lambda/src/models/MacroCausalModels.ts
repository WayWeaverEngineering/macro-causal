export interface MacroEconomicData {
  gdp: number;
  inflation: number;
  unemployment: number;
  interest_rate: number;
  exchange_rate: number;
  fiscal_deficit: number;
  trade_balance: number;
  date: string;
}

export interface PolicyIntervention {
  type: 'monetary' | 'fiscal' | 'trade' | 'regulatory';
  description: string;
  magnitude: number;
  start_date: string;
  end_date?: string;
  target_sectors?: string[];
}

export interface CausalAnalysisContext {
  treatment: PolicyIntervention;
  outcome: MacroEconomicData;
  confounders: MacroEconomicData[];
  pre_treatment_period: string[];
  post_treatment_period: string[];
  control_group?: MacroEconomicData[];
}

export interface MarketRegime {
  regime_id: number;
  regime_name: string;
  characteristics: {
    volatility: number;
    trend: 'bullish' | 'bearish' | 'sideways';
    correlation_structure: Record<string, number>;
    risk_level: 'low' | 'medium' | 'high';
  };
  probability: number;
  duration_estimate: number;
}

export interface UncertaintyMetrics {
  confidence_interval: [number, number];
  standard_error: number;
  p_value: number;
  effect_size: number;
  reliability_score: number;
  robustness_checks: {
    placebo_test: boolean;
    sensitivity_analysis: boolean;
    cross_validation: boolean;
  };
}

export interface MacroCausalInsight {
  insight_type: 'treatment_effect' | 'regime_identification' | 'uncertainty_quantification';
  summary: string;
  magnitude: number;
  direction: 'positive' | 'negative' | 'neutral';
  confidence: number;
  policy_implications: string[];
  limitations: string[];
  data_sources: string[];
  methodology: string;
}
