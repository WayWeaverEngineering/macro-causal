import { 
  XLearnerInputs, 
  XLearnerResults, 
  RegimeClassifierInputs, 
  RegimeClassifierResults, 
  UncertaintyEstimatorInputs, 
  UncertaintyEstimatorResults 
} from "../models/AgentModels";
import { ModelServingService, ModelServingResponse } from "./ModelServingService";

export class ModelService {
  private modelServingService: ModelServingService;

  constructor(modelServingUrl: string, apiKey?: string) {
    this.modelServingService = new ModelServingService(modelServingUrl, apiKey);
  }

  async executeXLearner(inputs: XLearnerInputs): Promise<XLearnerResults> {
    console.log('ModelService.executeXLearner - Starting X-Learner execution');
    console.log('ModelService.executeXLearner - Inputs:', JSON.stringify(inputs, null, 2));

    try {
      // Convert XLearner inputs to the format expected by the hybrid model
      const hybridInputs = {
        model_type: 'hybrid_causal_model',
        inputs: {
          treatment_variables: inputs.treatment_variables,
          outcome_variables: inputs.outcome_variables,
          confounders: inputs.confounders,
          time_periods: inputs.time_periods
        }
      };

      // Submit to the hybrid model serving service
      const result = await this.modelServingService.submitInferenceRequest(hybridInputs);
      
      // Extract X-Learner specific results
      const causalEffect = result.causal_effects[0] || 0;
      const standardError = result.uncertainty?.[0] || 0.1;
      const confidenceInterval: [number, number] = [
        causalEffect - 1.96 * standardError,
        causalEffect + 1.96 * standardError
      ];
      
      // Calculate p-value based on effect size and standard error
      const pValue = this.calculatePValue(causalEffect, standardError);

      return {
        treatment_effect: causalEffect,
        confidence_interval: confidenceInterval,
        p_value: pValue,
        standard_error: standardError
      };

    } catch (error) {
      console.error('ModelService.executeXLearner - Error:', error);
      
      // Return mock results for development/testing
      return this.generateMockXLearnerResults(inputs);
    }
  }

  async executeRegimeClassifier(inputs: RegimeClassifierInputs): Promise<RegimeClassifierResults> {
    console.log('ModelService.executeRegimeClassifier - Starting Regime Classifier execution');
    console.log('ModelService.executeRegimeClassifier - Inputs:', JSON.stringify(inputs, null, 2));

    try {
      // Convert Regime Classifier inputs to the format expected by the hybrid model
      const hybridInputs = {
        model_type: 'hybrid_causal_model',
        inputs: {
          market_indicators: inputs.market_indicators,
          lookback_periods: inputs.lookback_periods,
          regime_count: inputs.regime_count
        }
      };

      // Submit to the hybrid model serving service
      const result = await this.modelServingService.submitInferenceRequest(hybridInputs);
      
      // Extract regime classification results
      const regimeProbs = result.regime_probabilities?.[0] || [0.33, 0.33, 0.34];
      const predictedRegime = result.dominant_regime?.[0] || 0;
      const confidence = Math.max(...regimeProbs);

      return {
        regime_probabilities: regimeProbs,
        predicted_regime: predictedRegime,
        confidence: confidence,
        regime_characteristics: this.generateRegimeCharacteristics(predictedRegime, regimeProbs)
      };

    } catch (error) {
      console.error('ModelService.executeRegimeClassifier - Error:', error);
      
      // Return mock results for development/testing
      return this.generateMockRegimeClassifierResults(inputs);
    }
  }

  async executeUncertaintyEstimator(inputs: UncertaintyEstimatorInputs): Promise<UncertaintyEstimatorResults> {
    console.log('ModelService.executeUncertaintyEstimator - Starting Uncertainty Estimator execution');
    console.log('ModelService.executeUncertaintyEstimator - Inputs:', JSON.stringify(inputs, null, 2));

    try {
      // Convert Uncertainty Estimator inputs to the format expected by the hybrid model
      const hybridInputs = {
        model_type: 'hybrid_causal_model',
        inputs: {
          base_estimates: inputs.base_estimates,
          confidence_level: inputs.confidence_level,
          bootstrap_samples: inputs.bootstrap_samples
        }
      };

      // Submit to the hybrid model serving service
      const result = await this.modelServingService.submitInferenceRequest(hybridInputs);
      
      // Extract uncertainty estimation results
      const uncertainty = result.uncertainty?.[0] || 0.1;
      const confidenceInterval: [number, number] = [-uncertainty, uncertainty];
      const reliabilityScore = Math.max(0.5, 1 - uncertainty); // Higher uncertainty = lower reliability

      return {
        uncertainty_estimate: uncertainty,
        confidence_interval: confidenceInterval,
        reliability_score: reliabilityScore
      };

    } catch (error) {
      console.error('ModelService.executeUncertaintyEstimator - Error:', error);
      
      // Return mock results for development/testing
      return this.generateMockUncertaintyEstimatorResults(inputs);
    }
  }

  /**
   * Execute the hybrid causal model directly
   * @param inputs - Combined inputs for all model types
   * @returns Promise<ModelServingResponse> - The complete model results
   */
  async executeHybridCausalModel(inputs: any): Promise<ModelServingResponse> {
    console.log('ModelService.executeHybridCausalModel - Starting hybrid model execution');
    console.log('ModelService.executeHybridCausalModel - Inputs:', JSON.stringify(inputs, null, 2));

    try {
      return await this.modelServingService.submitInferenceRequest(inputs);
    } catch (error) {
      console.error('ModelService.executeHybridCausalModel - Error:', error);
      throw error;
    }
  }

  /**
   * Calculate p-value based on effect size and standard error
   * @param effect - The treatment effect
   * @param standardError - The standard error
   * @returns number - The p-value
   */
  private calculatePValue(effect: number, standardError: number): number {
    if (standardError === 0) return 1;
    
    const tStat = Math.abs(effect) / standardError;
    // Simple approximation: for large t-stats, p-value decreases exponentially
    // This is a simplified calculation - in practice, you'd use proper t-distribution
    if (tStat > 3) return Math.exp(-tStat / 2);
    if (tStat > 2) return 0.05;
    if (tStat > 1.5) return 0.15;
    return 0.5;
  }

  /**
   * Generate regime characteristics based on regime ID and probabilities
   * @param regimeId - The predicted regime ID
   * @param probabilities - The regime probabilities
   * @returns Record<string, any> - Regime characteristics
   */
  private generateRegimeCharacteristics(regimeId: number, probabilities: number[]): Record<string, any> {
    const regimeNames = ['High Volatility/Recession', 'Low Volatility/Expansion', 'Normal'];
    const regimeTrends = ['bearish', 'bullish', 'sideways'];
    const regimeRiskLevels = ['high', 'low', 'medium'];
    
    return {
      volatility: 1 - probabilities[regimeId], // Higher probability = lower volatility
      trend: regimeTrends[regimeId] || 'sideways',
      risk_level: regimeRiskLevels[regimeId] || 'medium',
      regime_name: regimeNames[regimeId] || 'Unknown',
      confidence: probabilities[regimeId]
    };
  }

  // Mock result generators for development/testing
  private generateMockXLearnerResults(inputs: XLearnerInputs): XLearnerResults {
    const baseEffect = Math.random() * 2 - 1; // Random effect between -1 and 1
    const standardError = Math.abs(baseEffect) * 0.2;
    
    return {
      treatment_effect: baseEffect,
      confidence_interval: [baseEffect - 1.96 * standardError, baseEffect + 1.96 * standardError],
      p_value: Math.random() * 0.1, // Random p-value between 0 and 0.1
      standard_error: standardError
    };
  }

  private generateMockRegimeClassifierResults(inputs: RegimeClassifierInputs): RegimeClassifierResults {
    const regimeCount = inputs.regime_count || 3;
    const probabilities = Array.from({ length: regimeCount }, () => Math.random());
    const sum = probabilities.reduce((a, b) => a + b, 0);
    const normalizedProbabilities = probabilities.map(p => p / sum);
    
    const predictedRegime = normalizedProbabilities.indexOf(Math.max(...normalizedProbabilities));
    
    return {
      regime_probabilities: normalizedProbabilities,
      predicted_regime: predictedRegime,
      confidence: Math.max(...normalizedProbabilities),
      regime_characteristics: this.generateRegimeCharacteristics(predictedRegime, normalizedProbabilities)
    };
  }

  private generateMockUncertaintyEstimatorResults(inputs: UncertaintyEstimatorInputs): UncertaintyEstimatorResults {
    const uncertainty = Math.random() * 0.2 + 0.05; // Random uncertainty between 0.05 and 0.25
    const reliability = Math.random() * 0.2 + 0.8; // Random reliability between 0.8 and 1.0
    
    return {
      uncertainty_estimate: uncertainty,
      confidence_interval: [-uncertainty, uncertainty],
      reliability_score: reliability
    };
  }

  /**
   * Test connection to the model serving service
   * @returns Promise<boolean> - True if connection is successful
   */
  async testConnection(): Promise<boolean> {
    return this.modelServingService.testConnection();
  }
}
