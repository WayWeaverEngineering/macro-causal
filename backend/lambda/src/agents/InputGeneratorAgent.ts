import { ChatOpenAI } from "@langchain/openai";
import { AgentState, ExecutionStep, ModelInputs } from "../models/AgentModels";

export class InputGeneratorAgent {
  private model: ChatOpenAI;

  constructor(model: ChatOpenAI) {
    this.model = model;
  }

  async invoke(state: AgentState): Promise<Partial<AgentState>> {
    const step: ExecutionStep = {
      stepId: "input-generation",
      stepName: "Generating Model Inputs",
      description: "Converting natural language query to structured inputs for ML models",
      status: "in_progress",
      startTime: new Date()
    };

    try {
      if (!state.promptAnalysis) {
        throw new Error("No prompt analysis available");
      }

      // Generate structured inputs based on analysis type
      const generatedInputs = await this.generateModelInputs(
        state.userQuery,
        state.promptAnalysis
      );
      
      step.status = "completed";
      step.endTime = new Date();
      step.metadata = {
        analysisType: state.promptAnalysis.analysisType,
        inputsGenerated: Object.keys(generatedInputs).length
      };

      return {
        generatedInputs,
        currentStep: step,
        executionSteps: this.updateExecutionSteps(state.executionSteps, step)
      };

    } catch (error) {
      step.status = "failed";
      step.endTime = new Date();
      step.error = error instanceof Error ? error.message : "Unknown error";

      return {
        currentStep: step,
        executionSteps: this.updateExecutionSteps(state.executionSteps, step),
        error: step.error
      };
    }
  }

  private async generateModelInputs(
    userQuery: string,
    promptAnalysis: any
  ): Promise<ModelInputs> {
    console.log('InputGeneratorAgent.generateModelInputs - Starting input generation');
    console.log('InputGeneratorAgent.generateModelInputs - Analysis type:', promptAnalysis.analysisType);
    
    const prompt = `
Convert this natural language query to structured inputs for ML models:

QUERY: "${userQuery}"
ANALYSIS TYPE: ${promptAnalysis.analysisType}

AVAILABLE MODELS:
1. X-Learner: Treatment effect estimation for policy interventions
2. Regime Classifier: Market state identification and regime classification
3. Uncertainty Estimator: Confidence intervals and uncertainty quantification

Generate structured inputs for each relevant model based on the analysis type:

X-LEARNER INPUTS (for treatment_effect analysis):
{
  "treatment_variables": ["list of treatment variables"],
  "outcome_variables": ["list of outcome variables"],
  "confounders": ["list of confounding variables"],
  "time_periods": {"start": "YYYY-MM", "end": "YYYY-MM"}
}

REGIME CLASSIFIER INPUTS (for regime_classification analysis):
{
  "market_indicators": ["list of market indicators"],
  "lookback_periods": number,
  "regime_count": number
}

UNCERTAINTY ESTIMATOR INPUTS (for uncertainty_estimation analysis):
{
  "base_estimates": ["list of base estimates"],
  "confidence_level": number,
  "bootstrap_samples": number
}

Only include the models that are relevant to the analysis type. If the analysis type is "other", include all models.

RESPONSE (JSON only):
`;

    try {
      const response = await this.model.invoke(prompt);
      console.log('InputGeneratorAgent.generateModelInputs - Model invocation successful');
      
      const content = response.content as string;
      console.log('InputGeneratorAgent.generateModelInputs - Response content length:', content.length);
      
      try {
        // Extract JSON from response
        const jsonMatch = content.match(/\{[\s\S]*\}/);
        if (!jsonMatch) {
          throw new Error("No JSON found in response");
        }
        
        const result = JSON.parse(jsonMatch[0]) as ModelInputs;
        console.log('InputGeneratorAgent.generateModelInputs - JSON parsed successfully');
        
        // Validate the generated inputs
        return this.validateAndCleanInputs(result);
        
      } catch (error) {
        console.log('InputGeneratorAgent.generateModelInputs - JSON parsing failed:', error);
        // Fallback inputs
        return this.generateFallbackInputs(promptAnalysis.analysisType);
      }
    } catch (error) {
      console.log('InputGeneratorAgent.generateModelInputs - Model invocation failed:', error);
      // Fallback inputs
      return this.generateFallbackInputs(promptAnalysis.analysisType);
    }
  }

  private validateAndCleanInputs(inputs: ModelInputs): ModelInputs {
    const cleaned: ModelInputs = {};

    if (inputs.xLearner) {
      cleaned.xLearner = {
        treatment_variables: inputs.xLearner.treatment_variables || [],
        outcome_variables: inputs.xLearner.outcome_variables || [],
        confounders: inputs.xLearner.confounders || [],
        time_periods: inputs.xLearner.time_periods || { start: "2020-01", end: "2024-01" }
      };
    }

    if (inputs.regimeClassifier) {
      cleaned.regimeClassifier = {
        market_indicators: inputs.regimeClassifier.market_indicators || [],
        lookback_periods: inputs.regimeClassifier.lookback_periods || 12,
        regime_count: inputs.regimeClassifier.regime_count || 3
      };
    }

    if (inputs.uncertaintyEstimator) {
      cleaned.uncertaintyEstimator = {
        base_estimates: inputs.uncertaintyEstimator.base_estimates || [],
        confidence_level: inputs.uncertaintyEstimator.confidence_level || 0.95,
        bootstrap_samples: inputs.uncertaintyEstimator.bootstrap_samples || 1000
      };
    }

    return cleaned;
  }

  private generateFallbackInputs(analysisType: string): ModelInputs {
    const fallback: ModelInputs = {};

    if (analysisType === 'treatment_effect' || analysisType === 'other') {
      fallback.xLearner = {
        treatment_variables: ['policy_intervention'],
        outcome_variables: ['gdp_growth', 'inflation_rate'],
        confounders: ['interest_rate', 'exchange_rate'],
        time_periods: { start: "2020-01", end: "2024-01" }
      };
    }

    if (analysisType === 'regime_classification' || analysisType === 'other') {
      fallback.regimeClassifier = {
        market_indicators: ['volatility_index', 'trend_indicator'],
        lookback_periods: 12,
        regime_count: 3
      };
    }

    if (analysisType === 'uncertainty_estimation' || analysisType === 'other') {
      fallback.uncertaintyEstimator = {
        base_estimates: ['treatment_effect_estimate'],
        confidence_level: 0.95,
        bootstrap_samples: 1000
      };
    }

    return fallback;
  }

  private updateExecutionSteps(
    currentSteps: ExecutionStep[], 
    updatedStep: ExecutionStep
  ): ExecutionStep[] {
    return currentSteps.map(step => 
      step.stepId === updatedStep.stepId ? updatedStep : step
    );
  }
}
