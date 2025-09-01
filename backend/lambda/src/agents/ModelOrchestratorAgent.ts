import { AgentState, ExecutionStep, ModelInputs, ModelResults } from "../models/AgentModels";
import { ModelService } from "../services/ModelService";

export class ModelOrchestratorAgent {
  private modelService: ModelService;

  constructor(modelService: ModelService) {
    this.modelService = modelService;
  }

  async invoke(state: AgentState): Promise<Partial<AgentState>> {
    const step: ExecutionStep = {
      stepId: "model-execution",
      stepName: "Executing ML Models",
      description: "Running ML models with generated inputs to produce analysis results",
      status: "in_progress",
      startTime: new Date()
    };

    try {
      if (!state.generatedInputs) {
        throw new Error("No generated inputs available");
      }

      // Execute models based on available inputs
      const modelResults = await this.executeModels(state.generatedInputs);
      
      step.status = "completed";
      step.endTime = new Date();
      step.metadata = {
        modelsExecuted: Object.keys(modelResults).length,
        executionTime: Date.now() - step.startTime!.getTime()
      };

      return {
        modelResults,
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

  private async executeModels(inputs: ModelInputs): Promise<ModelResults> {
    console.log('ModelOrchestratorAgent.executeModels - Starting model execution');
    console.log('ModelOrchestratorAgent.executeModels - Available inputs:', Object.keys(inputs));
    
    const results: ModelResults = {};

    try {
      // Execute X-Learner if applicable
      if (inputs.xLearner) {
        console.log('ModelOrchestratorAgent.executeModels - Executing X-Learner');
        results.xLearner = await this.modelService.executeXLearner(inputs.xLearner);
        console.log('ModelOrchestratorAgent.executeModels - X-Learner completed');
      }

      // Execute Regime Classifier if applicable
      if (inputs.regimeClassifier) {
        console.log('ModelOrchestratorAgent.executeModels - Executing Regime Classifier');
        results.regimeClassifier = await this.modelService.executeRegimeClassifier(inputs.regimeClassifier);
        console.log('ModelOrchestratorAgent.executeModels - Regime Classifier completed');
      }

      // Execute Uncertainty Estimator if applicable
      if (inputs.uncertaintyEstimator) {
        console.log('ModelOrchestratorAgent.executeModels - Executing Uncertainty Estimator');
        results.uncertaintyEstimator = await this.modelService.executeUncertaintyEstimator(inputs.uncertaintyEstimator);
        console.log('ModelOrchestratorAgent.executeModels - Uncertainty Estimator completed');
      }

      console.log('ModelOrchestratorAgent.executeModels - All models executed successfully');
      return results;

    } catch (error) {
      console.error('ModelOrchestratorAgent.executeModels - Error during model execution:', error);
      
      // Return partial results if some models succeeded
      if (Object.keys(results).length > 0) {
        console.log('ModelOrchestratorAgent.executeModels - Returning partial results');
        return results;
      }
      
      throw error;
    }
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
