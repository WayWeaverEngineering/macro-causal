import { CausalAnalysisResult } from '../../models/analysis';

type BackendResult = any;

const clamp01 = (value: number): number => {
  if (Number.isNaN(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
};

export const mapBackendResultToCausalAnalysis = (result: BackendResult | null | undefined): CausalAnalysisResult | null => {
  if (!result || !result.analysis) return null;

  const analysis = result.analysis ?? {};
  const modelResults = analysis.model_results ?? {};

  const effect: number = Array.isArray(modelResults.causal_effects) && modelResults.causal_effects.length > 0
    ? Number(modelResults.causal_effects[0])
    : 0;

  const uncertaintyRaw: number = Array.isArray(modelResults.uncertainty) && modelResults.uncertainty.length > 0
    ? Number(modelResults.uncertainty[0])
    : 0;

  const uncertainty = clamp01(uncertaintyRaw);
  const ciHalfWidth = Math.abs(effect) * 0.25 + uncertainty; // heuristic when CI not provided
  const confidenceInterval: [number, number] = [effect - ciHalfWidth, effect + ciHalfWidth];

  const pValue = Math.max(0, Math.min(1, 1 - (1 - uncertainty))); // placeholder derived from uncertainty

  const significance: 'high' | 'medium' | 'low' =
    uncertainty < 0.1 ? 'high' : uncertainty < 0.2 ? 'medium' : 'low';

  const economicSignificance: 'high' | 'medium' | 'low' =
    Math.abs(effect) >= 0.02 ? 'high' : Math.abs(effect) >= 0.01 ? 'medium' : 'low';

  const direction: 'positive' | 'negative' | 'neutral' =
    effect > 0 ? 'positive' : effect < 0 ? 'negative' : 'neutral';

  const regimeProbabilitiesArray: number[] =
    Array.isArray(modelResults.regime_probabilities) && modelResults.regime_probabilities.length > 0
      ? (modelResults.regime_probabilities[0] as number[]).map((x) => clamp01(Number(x)))
      : [];

  const currentRegime: number = Array.isArray(modelResults.dominant_regime) && modelResults.dominant_regime.length > 0
    ? Number(modelResults.dominant_regime[0])
    : 0;

  const regimeNames: string[] = regimeProbabilitiesArray.map((_, idx) => `Regime ${idx}`);

  const mapped: CausalAnalysisResult = {
    summary: String(analysis.summary ?? ''),
    keyInsights: Array.isArray(analysis.key_insights) ? analysis.key_insights.map(String) : [],
    causalEffect: {
      effect,
      confidenceInterval,
      pValue,
      significance,
      economicSignificance,
      direction,
    },
    regimeAnalysis: {
      currentRegime,
      regimeProbabilities: regimeProbabilitiesArray,
      regimeNames,
      regimeEffects: {},
      regimeFeatures: {},
    },
    uncertainty: {
      uncertainty,
      confidence: clamp01(1 - uncertainty),
      reliability: uncertainty < 0.1 ? 'high' : uncertainty < 0.2 ? 'medium' : 'low',
      factors: [],
    },
    methodology: String(analysis.methodology ?? ''),
    limitations: Array.isArray(analysis.limitations) ? analysis.limitations.map(String) : [],
  };

  return mapped;
};


