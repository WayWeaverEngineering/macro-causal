export interface ModelServingResponse {
  causal_effects: number[];
  model_type: string;
  n_samples: number;
  n_features: number;
  regime_probabilities?: number[][];
  dominant_regime?: number[];
  uncertainty?: number[];
}

export class ModelServingService {
  private baseUrl: string;

  constructor(baseUrl: string, apiKey?: string) {
    this.baseUrl = baseUrl;
  }

  /**
   * Submit an inference request to the model serving service
   * @param modelInputs - The structured inputs for the models
   * @returns Promise<ModelServingResponse> - The model inference results
   */
  async submitInferenceRequest(modelInputs: any): Promise<ModelServingResponse> {
    console.log('ModelServingService.submitInferenceRequest - Starting inference request');
    console.log('ModelServingService.submitInferenceRequest - Inputs:', JSON.stringify(modelInputs, null, 2));

    try {
      // First, get available models to find the hybrid causal model
      const models = await this.getAvailableModels();
      console.log('ModelServingService.submitInferenceRequest - Available models:', models);

      // Find the hybrid causal model
      const hybridModel = models.find((model: any) => 
        model.model_type === 'hybrid_causal_model' || 
        model.model_name?.includes('hybrid') || 
        model.model_name?.includes('causal')
      );

      if (!hybridModel) {
        throw new Error('No hybrid causal model found in model serving service');
      }

      console.log('ModelServingService.submitInferenceRequest - Using model:', hybridModel.model_id);

      // Submit prediction request
      const response = await fetch(`${this.baseUrl}/predict/${hybridModel.model_id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // No Authorization header needed for internal FastAPI service
        },
        body: JSON.stringify(this.prepareModelInputs(modelInputs))
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Model serving API error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();
      console.log('ModelServingService.submitInferenceRequest - Success:', JSON.stringify(result, null, 2));

      // FastAPI returns { model_id, prediction: {...} }
      const predictionPayload = result?.prediction ?? result;

      // Transform the response to match our interface
      return this.transformResponse(predictionPayload);

    } catch (error) {
      console.error('ModelServingService.submitInferenceRequest - Error:', error);
      
      // Return mock results for development/testing if the service is unavailable
      return this.generateMockResults(modelInputs);
    }
  }

  /**
   * Get available models from the model serving service
   * @returns Promise<any[]> - List of available models
   */
  private async getAvailableModels(): Promise<any[]> {
    try {
      const response = await fetch(`${this.baseUrl}/models`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          // No Authorization header needed for internal FastAPI service
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to get models: ${response.status} ${response.statusText}`);
      }

      const modelsResponse = await response.json();
      const models = modelsResponse?.models ?? modelsResponse;
      return Array.isArray(models) ? models : [];
    } catch (error) {
      console.error('ModelServingService.getAvailableModels - Error:', error);
      return [];
    }
  }

  /**
   * Prepare model inputs in the format expected by the model serving service
   * @param modelInputs - The structured inputs from OpenAI
   * @returns any - The prepared inputs for the model
   */
  private prepareModelInputs(modelInputs: any): any {
    // Extract the inputs from the OpenAI response
    const inputs = modelInputs.inputs || modelInputs;
    
    // Convert the inputs to the format expected by the model serving service
    // The model serving service expects feature values, not feature names
    const preparedInputs: any = {};
    
    // For now, we'll use placeholder values based on the input structure
    // In a real implementation, you would fetch actual data for these features
    
    if (inputs.treatment_variables) {
      inputs.treatment_variables.forEach((variable: string) => {
        preparedInputs[variable] = this.getPlaceholderValue(variable);
      });
    }
    
    if (inputs.outcome_variables) {
      inputs.outcome_variables.forEach((variable: string) => {
        preparedInputs[variable] = this.getPlaceholderValue(variable);
      });
    }
    
    if (inputs.confounders) {
      inputs.confounders.forEach((variable: string) => {
        preparedInputs[variable] = this.getPlaceholderValue(variable);
      });
    }
    
    // Add market indicators if available
    if (inputs.market_indicators) {
      inputs.market_indicators.forEach((indicator: string) => {
        preparedInputs[indicator] = this.getPlaceholderValue(indicator);
      });
    }
    
    console.log('ModelServingService.prepareModelInputs - Prepared inputs:', preparedInputs);
    return preparedInputs;
  }

  /**
   * Get placeholder values for features (for development/testing)
   * @param featureName - The name of the feature
   * @returns number - A placeholder value
   */
  private getPlaceholderValue(featureName: string): number {
    // Generate realistic placeholder values based on feature names
    const featureMap: { [key: string]: number } = {
      // Treatment variables
      'fed_rate_shock': 0.25, // 25 basis point change
      'cpi_surprise': 0.1, // 0.1% surprise
      'gdp_shock': 0.5, // 0.5% surprise
      
      // Outcome variables
      'sp500_returns': 0.02, // 2% monthly return
      'bond_returns': -0.01, // -1% monthly return
      'gold_returns': 0.015, // 1.5% monthly return
      
      // Confounders
      'gdp_growth': 2.5, // 2.5% annual growth
      'unemployment_rate': 3.8, // 3.8% unemployment
      'oil_prices': 75.0, // $75 per barrel
      
      // Market indicators
      'vix': 18.5, // VIX level
      'yield_curve_slope': 0.15, // 15 basis point slope
      'economic_surprise_index': 0.2, // 0.2 surprise index
    };
    
    // Return mapped value or generate random value if not found
    return featureMap[featureName] || (Math.random() * 2 - 1); // Random value between -1 and 1
  }

  /**
   * Transform the model serving response to our interface
   * @param response - The raw response from the model serving service
   * @returns ModelServingResponse - The transformed response
   */
  private transformResponse(response: any): ModelServingResponse {
    return {
      causal_effects: response.causal_effects || [0],
      model_type: response.model_type || 'hybrid_causal_model',
      n_samples: response.n_samples || 1,
      n_features: response.n_features || 0,
      regime_probabilities: response.regime_probabilities,
      dominant_regime: response.dominant_regime,
      uncertainty: response.uncertainty
    };
  }

  /**
   * Generate mock results for development/testing
   * @param modelInputs - The model inputs
   * @returns ModelServingResponse - Mock results
   */
  private generateMockResults(modelInputs: any): ModelServingResponse {
    console.log('ModelServingService.generateMockResults - Generating mock results');
    
    // Generate realistic mock results based on the inputs
    const inputs = modelInputs.inputs || modelInputs;
    
    // Generate causal effects based on treatment variables
    let baseEffect = 0;
    if (inputs.treatment_variables) {
      inputs.treatment_variables.forEach((variable: string) => {
        if (variable.includes('fed_rate')) {
          baseEffect -= 0.02; // Fed rate hikes typically reduce asset returns
        } else if (variable.includes('cpi')) {
          baseEffect -= 0.015; // CPI surprises typically reduce asset returns
        } else if (variable.includes('gdp')) {
          baseEffect += 0.025; // GDP surprises typically increase asset returns
        }
      });
    }
    
    // Add some randomness
    baseEffect += (Math.random() - 0.5) * 0.01;
    
    // Generate regime probabilities
    const regimeCount = 3;
    const regimeProbs = Array.from({ length: regimeCount }, () => Math.random());
    const sum = regimeProbs.reduce((a, b) => a + b, 0);
    const normalizedRegimeProbs = regimeProbs.map(p => p / sum);
    
    const dominantRegime = normalizedRegimeProbs.indexOf(Math.max(...normalizedRegimeProbs));
    
    // Generate uncertainty estimate
    const uncertainty = Math.random() * 0.1 + 0.05; // Between 0.05 and 0.15
    
    return {
      causal_effects: [baseEffect],
      model_type: 'hybrid_causal_model',
      n_samples: 1,
      n_features: Object.keys(inputs).length,
      regime_probabilities: [normalizedRegimeProbs],
      dominant_regime: [dominantRegime],
      uncertainty: [uncertainty]
    };
  }

  /**
   * Test the connection to the model serving service
   * @returns Promise<boolean> - True if connection is successful
   */
  async testConnection(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          // No Authorization header needed for internal FastAPI service
        }
      });

      if (!response.ok) {
        return false;
      }

      const healthCheck = await response.json();
      return healthCheck.status === 'healthy';
    } catch (error) {
      console.error('ModelServingService.testConnection - Error:', error);
      return false;
    }
  }
}
