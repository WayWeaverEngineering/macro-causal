import { AgentState, QueryResponse } from "../models/AgentModels";
import { QueryAnalyzerAgent } from "../agents/QueryAnalyzerAgent";
import { InputGeneratorAgent } from "../agents/InputGeneratorAgent";
import { ModelOrchestratorAgent } from "../agents/ModelOrchestratorAgent";
import { ResponseGeneratorAgent } from "../agents/ResponseGeneratorAgent";
import { OpenAIService } from "../services/OpenAIService";
import { ModelService } from "../services/ModelService";

export class MacroCausalOrchestrator {
  private agents: {
    queryAnalyzer: QueryAnalyzerAgent;
    inputGenerator: InputGeneratorAgent;
    modelOrchestrator: ModelOrchestratorAgent;
    responseGenerator: ResponseGeneratorAgent;
  };

  constructor(openAIService: OpenAIService, modelService: ModelService) {
    const model = openAIService.getModel();
    
    this.agents = {
      queryAnalyzer: new QueryAnalyzerAgent(model),
      inputGenerator: new InputGeneratorAgent(model),
      modelOrchestrator: new ModelOrchestratorAgent(modelService),
      responseGenerator: new ResponseGeneratorAgent(model)
    };
  }

  async executeQuery(userQuery: string): Promise<QueryResponse> {
    const startTime = Date.now();
    console.log('MacroCausalOrchestrator.executeQuery - Starting query execution');
    console.log('MacroCausalOrchestrator.executeQuery - User query:', userQuery);

    let state: AgentState = this.createInitialState(userQuery);

    try {
      // Step 1: Analyze query scope
      console.log('MacroCausalOrchestrator.executeQuery - Step 1: Analyzing query scope');
      const scopeResult = await this.agents.queryAnalyzer.invoke(state);
      state = { ...state, ...scopeResult };

      if (!state.isInScope) {
        console.log('MacroCausalOrchestrator.executeQuery - Query out of scope, generating response');
        return this.generateOutOfScopeResponse(state, startTime);
      }

      // Step 2: Generate model inputs
      console.log('MacroCausalOrchestrator.executeQuery - Step 2: Generating model inputs');
      const inputResult = await this.agents.inputGenerator.invoke(state);
      state = { ...state, ...inputResult };

      if (state.error) {
        throw new Error(`Input generation failed: ${state.error}`);
      }

      // Step 3: Execute models
      console.log('MacroCausalOrchestrator.executeQuery - Step 3: Executing ML models');
      const modelResult = await this.agents.modelOrchestrator.invoke(state);
      state = { ...state, ...modelResult };

      if (state.error) {
        throw new Error(`Model execution failed: ${state.error}`);
      }

      // Step 4: Generate final response
      console.log('MacroCausalOrchestrator.executeQuery - Step 4: Generating final response');
      const responseResult = await this.agents.responseGenerator.invoke(state);
      state = { ...state, ...responseResult };

      if (state.error) {
        throw new Error(`Response generation failed: ${state.error}`);
      }

      const executionTime = Date.now() - startTime;
      console.log('MacroCausalOrchestrator.executeQuery - Query execution completed successfully');

      return {
        success: true,
        response: state.finalResponse || "Analysis completed successfully",
        modelResults: state.modelResults || undefined,
        metadata: {
          query: userQuery,
          analysisType: state.promptAnalysis?.analysisType || 'unknown',
          complexity: state.promptAnalysis?.complexity || 'unknown',
          executionTime,
          stepsCompleted: state.executionSteps.filter(s => s.status === "completed").length,
          totalSteps: state.executionSteps.length,
          timestamp: new Date().toISOString()
        }
      };

    } catch (error) {
      const executionTime = Date.now() - startTime;
      console.error('MacroCausalOrchestrator.executeQuery - Error during execution:', error);
      
      return this.handleError(error, state, executionTime);
    }
  }

  private createInitialState(userQuery: string): AgentState {
    return {
      userQuery,
      executionSteps: [],
      currentStep: null,
      metadata: {},
      isInScope: false,
      promptAnalysis: null,
      generatedInputs: null,
      modelResults: null,
      finalResponse: null,
      error: null
    };
  }

  private generateOutOfScopeResponse(state: AgentState, startTime: number): QueryResponse {
    const executionTime = Date.now() - startTime;
    
    return {
      success: false,
      response: "Your query is outside the scope of this macro-causal analysis system.",
      outOfScopeReason: state.promptAnalysis?.reasoning || "Query not suitable for causal inference analysis",
      metadata: {
        query: state.userQuery,
        analysisType: 'out_of_scope',
        complexity: 'none',
        executionTime,
        stepsCompleted: 1,
        totalSteps: 1,
        timestamp: new Date().toISOString()
      }
    };
  }

  private handleError(error: any, state: AgentState, executionTime: number): QueryResponse {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    
    return {
      success: false,
      error: errorMessage,
      metadata: {
        query: state.userQuery,
        analysisType: state.promptAnalysis?.analysisType || 'unknown',
        complexity: state.promptAnalysis?.complexity || 'unknown',
        executionTime,
        stepsCompleted: state.executionSteps.filter(s => s.status === "completed").length,
        totalSteps: state.executionSteps.length,
        timestamp: new Date().toISOString()
      }
    };
  }

  // Utility method to get execution status
  getExecutionStatus(state: AgentState) {
    return {
      currentStep: state.currentStep,
      totalSteps: state.executionSteps.length,
      completedSteps: state.executionSteps.filter(s => s.status === "completed").length,
      failedSteps: state.executionSteps.filter(s => s.status === "failed").length,
      isComplete: state.finalResponse !== null,
      hasError: state.error !== null
    };
  }
}
