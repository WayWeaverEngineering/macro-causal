import { ChatOpenAI } from "@langchain/openai";
import { AgentState, ExecutionStep, PromptAnalysis } from "../models/AgentModels";

export class QueryAnalyzerAgent {
  private model: ChatOpenAI;

  constructor(model: ChatOpenAI) {
    this.model = model;
  }

  async invoke(state: AgentState): Promise<Partial<AgentState>> {
    const step: ExecutionStep = {
      stepId: "query-analyzer",
      stepName: "Analyzing User Query",
      description: "Determining if the query is in scope for macro-causal analysis and planning execution steps",
      status: "in_progress",
      startTime: new Date()
    };

    try {
      // Analyze the prompt
      const analysis = await this.analyzePrompt(state.userQuery);
      
      // Create execution plan
      const executionSteps = this.createExecutionPlan(analysis);
      
      step.status = "completed";
      step.endTime = new Date();
      step.metadata = {
        analysis,
        executionSteps: executionSteps.length
      };

      return {
        executionSteps: [step, ...executionSteps],
        currentStep: step,
        isInScope: analysis.isInScope,
        metadata: {
          ...state.metadata,
          promptAnalysis: analysis
        }
      };

    } catch (error) {
      step.status = "failed";
      step.endTime = new Date();
      step.error = error instanceof Error ? error.message : "Unknown error";

      return {
        executionSteps: [step],
        currentStep: step,
        isInScope: false,
        error: step.error
      };
    }
  }

  private async analyzePrompt(userQuery: string): Promise<PromptAnalysis> {
    console.log('QueryAnalyzerAgent.analyzePrompt - Starting prompt analysis');
    console.log('QueryAnalyzerAgent.analyzePrompt - User query:', userQuery);
    
    const prompt = `
You are an expert macro-causal analysis assistant. Your task is to determine if a user query is in scope for macro-causal inference analysis.

SCOPE DEFINITION:
- Analysis must be related to macroeconomic causal inference
- Must involve treatment effects, policy analysis, or causal relationships
- Should be answerable using X-Learner, Regime Classifier, or Uncertainty Estimator models
- Topics include: monetary policy effects, fiscal policy impacts, trade policy analysis, financial regulation effects, etc.
- Queries about general market conditions without causal focus, or non-economic topics are OUT OF SCOPE

USER QUERY: "${userQuery}"

Analyze this query and respond with a JSON object containing:
{
  "isInScope": boolean,
  "reasoning": "detailed explanation of why in/out of scope",
  "analysisType": "treatment_effect|regime_classification|uncertainty_estimation|other",
  "complexity": "simple|moderate|complex",
  "requiredData": ["list of required data types"],
  "estimatedSteps": number
}

RESPONSE (JSON only):
`;

    console.log('QueryAnalyzerAgent.analyzePrompt - About to invoke model');
    
    try {
      const response = await this.model.invoke(prompt);
      console.log('QueryAnalyzerAgent.analyzePrompt - Model invocation successful');
      
      const content = response.content as string;
      console.log('QueryAnalyzerAgent.analyzePrompt - Response content length:', content.length);
      
      try {
        // Extract JSON from response
        const jsonMatch = content.match(/\{[\s\S]*\}/);
        if (!jsonMatch) {
          throw new Error("No JSON found in response");
        }
        
        const result = JSON.parse(jsonMatch[0]) as PromptAnalysis;
        console.log('QueryAnalyzerAgent.analyzePrompt - JSON parsed successfully');
        return result;
      } catch (error) {
        console.log('QueryAnalyzerAgent.analyzePrompt - JSON parsing failed:', error);
        // Fallback analysis
        return {
          isInScope: false,
          reasoning: "Unable to parse analysis response",
          analysisType: "other",
          complexity: "simple",
          requiredData: [],
          estimatedSteps: 1
        };
      }
    } catch (error) {
      console.log('QueryAnalyzerAgent.analyzePrompt - Model invocation failed:', error);
      // Fallback analysis
      return {
        isInScope: false,
        reasoning: "Unable to parse analysis response",
        analysisType: "other",
        complexity: "simple",
        requiredData: [],
        estimatedSteps: 1
      };
    }
  }

  private createExecutionPlan(analysis: PromptAnalysis): ExecutionStep[] {
    const steps: ExecutionStep[] = [];

    if (!analysis.isInScope) {
      steps.push({
        stepId: "out-of-scope-response",
        stepName: "Generate Out-of-Scope Response",
        description: "Inform user that their query is outside the system's scope",
        status: "pending"
      });
      return steps;
    }

    // Standard execution plan for in-scope queries
    steps.push(
      {
        stepId: "input-generation",
        stepName: "Generate Model Inputs",
        description: `Convert natural language query to structured inputs for ${analysis.analysisType} analysis`,
        status: "pending"
      },
      {
        stepId: "model-execution",
        stepName: "Execute ML Models",
        description: `Run ${analysis.analysisType} analysis using appropriate models`,
        status: "pending"
      },
      {
        stepId: "response-generation",
        stepName: "Generate Final Response",
        description: "Compile analysis results and generate comprehensive response",
        status: "pending"
      }
    );

    return steps;
  }
}
