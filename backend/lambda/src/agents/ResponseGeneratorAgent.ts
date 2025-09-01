import { ChatOpenAI } from "@langchain/openai";
import { AgentState, ExecutionStep, ModelResults } from "../models/AgentModels";

export class ResponseGeneratorAgent {
  private model: ChatOpenAI;

  constructor(model: ChatOpenAI) {
    this.model = model;
  }

  async invoke(state: AgentState): Promise<Partial<AgentState>> {
    const step: ExecutionStep = {
      stepId: "response-generation",
      stepName: "Generating Final Response",
      description: "Compiling analysis results and generating comprehensive response",
      status: "in_progress",
      startTime: new Date()
    };

    try {
      let finalResponse: string;

      if (!state.isInScope) {
        finalResponse = await this.generateOutOfScopeResponse(state);
      } else {
        finalResponse = await this.generateAnalysisResponse(state);
      }

      step.status = "completed";
      step.endTime = new Date();
      step.metadata = {
        responseType: state.isInScope ? "analysis" : "out-of-scope",
        hasModelResults: !!state.modelResults,
        stepsCompleted: state.executionSteps.filter(s => s.status === "completed").length
      };

      return {
        finalResponse,
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
        finalResponse: "An error occurred while generating the response.",
        error: step.error
      };
    }
  }

  private async generateOutOfScopeResponse(state: AgentState): Promise<string> {
    const prompt = `
You are an expert macro-causal analysis assistant. The user's query is outside the scope of this system.

USER QUERY: "${state.userQuery}"

Generate a helpful response that:
1. Politely explains why the query is out of scope
2. Suggests alternative approaches or rephrasing
3. Explains what types of queries ARE in scope
4. Provides examples of in-scope queries

SCOPE DEFINITION:
- Analysis must be related to macroeconomic causal inference
- Must involve treatment effects, policy analysis, or causal relationships
- Should be answerable using X-Learner, Regime Classifier, or Uncertainty Estimator models
- Topics: monetary policy effects, fiscal policy impacts, trade policy analysis, financial regulation effects, etc.

Respond with a helpful, professional message that guides the user toward in-scope queries.
`;

    const response = await this.model.invoke(prompt);
    return response.content as string;
  }

  private async generateAnalysisResponse(state: AgentState): Promise<string> {
    if (!state.modelResults) {
      return "Analysis could not be completed due to insufficient data or processing errors.";
    }

    const prompt = `
You are an expert macro-causal analyst. Generate a comprehensive response based on the analysis results.

USER QUERY: "${state.userQuery}"

MODEL RESULTS:
${this.formatModelResults(state.modelResults)}

ANALYSIS TYPE: ${state.promptAnalysis?.analysisType || 'unknown'}
COMPLEXITY: ${state.promptAnalysis?.complexity || 'unknown'}

Generate a professional, clear response that:
1. Directly answers the user's query
2. Explains the causal relationships found
3. Provides confidence levels and uncertainty estimates
4. Includes actionable insights
5. Acknowledges limitations
6. Uses business-friendly language

Format as a comprehensive business analysis report with clear sections.
`;

    const response = await this.model.invoke(prompt);
    return response.content as string;
  }

  private formatModelResults(modelResults: ModelResults): string {
    let formatted = '';

    if (modelResults.xLearner) {
      formatted += `
X-LEARNER RESULTS (Treatment Effect Analysis):
- Treatment Effect: ${modelResults.xLearner.treatment_effect}
- Confidence Interval: [${modelResults.xLearner.confidence_interval[0]}, ${modelResults.xLearner.confidence_interval[1]}]
- P-value: ${modelResults.xLearner.p_value}
- Standard Error: ${modelResults.xLearner.standard_error}
`;
    }

    if (modelResults.regimeClassifier) {
      formatted += `
REGIME CLASSIFIER RESULTS (Market State Analysis):
- Predicted Regime: ${modelResults.regimeClassifier.predicted_regime}
- Confidence: ${modelResults.regimeClassifier.confidence}
- Regime Probabilities: [${modelResults.regimeClassifier.regime_probabilities.join(', ')}]
- Regime Characteristics: ${JSON.stringify(modelResults.regimeClassifier.regime_characteristics, null, 2)}
`;
    }

    if (modelResults.uncertaintyEstimator) {
      formatted += `
UNCERTAINTY ESTIMATOR RESULTS (Confidence Analysis):
- Uncertainty Estimate: ${modelResults.uncertaintyEstimator.uncertainty_estimate}
- Confidence Interval: [${modelResults.uncertaintyEstimator.confidence_interval[0]}, ${modelResults.uncertaintyEstimator.confidence_interval[1]}]
- Reliability Score: ${modelResults.uncertaintyEstimator.reliability_score}
`;
    }

    return formatted || 'No model results available';
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
