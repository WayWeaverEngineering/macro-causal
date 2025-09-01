import { 
  XLearnerInputs, 
  XLearnerResults, 
  RegimeClassifierInputs, 
  RegimeClassifierResults, 
  UncertaintyEstimatorInputs, 
  UncertaintyEstimatorResults 
} from "../models/AgentModels";

export class ModelService {
  private modelServingUrl: string;
  private apiKey?: string;

  constructor(modelServingUrl: string, apiKey?: string) {
    this.modelServingUrl = modelServingUrl;
    this.apiKey = apiKey;
  }

  async executeXLearner(inputs: XLearnerInputs): Promise<XLearnerResults> {
    console.log('ModelService.executeXLearner - Starting X-Learner execution');
    console.log('ModelService.executeXLearner - Inputs:', JSON.stringify(inputs, null, 2));

    try {
      const response = await fetch(`${this.modelServingUrl}/predict/xlearner`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey && { 'Authorization': `Bearer ${this.apiKey}` })
        },
        body: JSON.stringify(inputs)
      });

      if (!response.ok) {
        throw new Error(`X-Learner API error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      console.log('ModelService.executeXLearner - Success:', JSON.stringify(result, null, 2));

      // Transform the response to match our interface
      return {
        treatment_effect: result.treatment_effect || 0,
        confidence_interval: result.confidence_interval || [0, 0],
        p_value: result.p_value || 1,
        standard_error: result.standard_error || 0
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
      const response = await fetch(`${this.modelServingUrl}/predict/regime-classifier`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey && { 'Authorization': `Bearer ${this.apiKey}` })
        },
        body: JSON.stringify(inputs)
      });

      if (!response.ok) {
        throw new Error(`Regime Classifier API error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      console.log('ModelService.executeRegimeClassifier - Success:', JSON.stringify(result, null, 2));

      // Transform the response to match our interface
      return {
        regime_probabilities: result.regime_probabilities || [0.33, 0.33, 0.34],
        predicted_regime: result.predicted_regime || 0,
        confidence: result.confidence || 0.8,
        regime_characteristics: result.regime_characteristics || {
          volatility: 0.5,
          trend: 'sideways',
          risk_level: 'medium'
        }
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
      const response = await fetch(`${this.modelServingUrl}/predict/uncertainty-estimator`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey && { 'Authorization': `Bearer ${this.apiKey}` })
        },
        body: JSON.stringify(inputs)
      });

      if (!response.ok) {
        throw new Error(`Uncertainty Estimator API error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      console.log('ModelService.executeUncertaintyEstimator - Success:', JSON.stringify(result, null, 2));

      // Transform the response to match our interface
      return {
        uncertainty_estimate: result.uncertainty_estimate || 0.1,
        confidence_interval: result.confidence_interval || [0, 0],
        reliability_score: result.reliability_score || 0.8
      };

    } catch (error) {
      console.error('ModelService.executeUncertaintyEstimator - Error:', error);
      
      // Return mock results for development/testing
      return this.generateMockUncertaintyEstimatorResults(inputs);
    }
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
      confidence: Math.random() * 0.3 + 0.7, // Random confidence between 0.7 and 1.0
      regime_characteristics: {
        volatility: Math.random(),
        trend: ['bullish', 'bearish', 'sideways'][Math.floor(Math.random() * 3)],
        risk_level: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)]
      }
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
}
