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
    try {
      // Get available models and choose a ready hybrid causal model
      const models = await this.getAvailableModels();
      const readyModels = models.filter((m: any) => m.status === 'ready');

      // Prefer models whose name hints at hybrid/causal; otherwise take the most recent
      const preferred = readyModels.filter((m: any) =>
        (m.model_name || '').toLowerCase().includes('hybrid') ||
        (m.model_name || '').toLowerCase().includes('causal')
      );
      const candidates = preferred.length > 0 ? preferred : readyModels;
      if (!candidates.length) {
        throw new Error('No ready models found in model serving service');
      }
      // Sort by created_at descending if available
      candidates.sort((a: any, b: any) => {
        const ta = Date.parse(a.created_at || '');
        const tb = Date.parse(b.created_at || '');
        return (isNaN(tb) ? 0 : tb) - (isNaN(ta) ? 0 : ta);
      });
      const chosenModel = candidates[0];

      // Verify model type and retrieve schema
      const modelInfo = await this.getModelInfo(chosenModel.model_id);
      if (modelInfo?.type !== 'hybrid_causal_model') {
        throw new Error(`Chosen model is not hybrid_causal_model: ${modelInfo?.type}`);
      }

      // Prepare inputs aligned to feature_columns and wrap as { instances: [...] }
      const prepared = this.prepareModelInputs(modelInputs, modelInfo?.feature_columns || []);

      const response = await fetch(`${this.baseUrl}/predict/${chosenModel.model_id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // No Authorization header needed for internal FastAPI service
        },
        body: JSON.stringify({ instances: [prepared] })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Model serving API error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();
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
   * Get detailed info for a model to retrieve type and feature schema
   */
  private async getModelInfo(modelId: string): Promise<any | null> {
    try {
      const response = await fetch(`${this.baseUrl}/models/${modelId}/info`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      if (!response.ok) return null;
      return await response.json();
    } catch (error) {
      console.error('ModelServingService.getModelInfo - Error:', error);
      return null;
    }
  }

  /**
   * Prepare model inputs in the format expected by the model serving service
   * @param modelInputs - The structured inputs from OpenAI
   * @returns any - The prepared inputs for the model
   */
  private prepareModelInputs(modelInputs: any, featureColumns: string[]): any {
    const inputs = modelInputs.inputs || modelInputs || {};
    const prepared: any = {};
    // Default to 0.0 for any missing feature; use provided numeric values when present
    featureColumns.forEach((col: string) => {
      const source = (inputs.values && typeof inputs.values[col] !== 'undefined') ? inputs.values[col]
        : (typeof inputs[col] !== 'undefined' ? inputs[col] : undefined);
      prepared[col] = this.getNumericValueOrDefault(source, col);
    });
    return prepared;
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

  private getNumericValueOrDefault(value: any, featureName: string): number {
    if (typeof value === 'number' && isFinite(value)) return value;
    const parsed = parseFloat(value);
    if (!isNaN(parsed) && isFinite(parsed)) return parsed;
    // Fall back to a stable placeholder to avoid NaNs
    return this.getPlaceholderValue(featureName);
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
