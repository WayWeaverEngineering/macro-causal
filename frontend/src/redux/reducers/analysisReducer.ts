import { createReducer } from '@reduxjs/toolkit';
import { AnalysisState } from '../states/types';
import { initialAnalysisState } from '../states/initialStates';
import * as actions from '../actions/analysisActions';

export const analysisReducer = createReducer<AnalysisState>(initialAnalysisState, (builder) => {
  builder
    .addCase(actions.startAnalysis, (state, action) => {
      state.currentQuery = action.payload;
      state.isExecuting = true;
      state.executionId = null;
      state.executionSteps = [];
      state.currentStep = null;
      state.analysis = null;
      state.error = null;
      state.isInScope = true;
      state.outOfScopeReason = undefined;
    })
    .addCase(actions.setExecutionId, (state, action) => {
      state.executionId = action.payload;
    })
    .addCase(actions.addExecutionStep, (state, action) => {
      state.executionSteps.push(action.payload);
    })
    .addCase(actions.updateExecutionStep, (state, action) => {
      const stepIndex = state.executionSteps.findIndex(step => step.stepId === action.payload.stepId);
      if (stepIndex !== -1) {
        state.executionSteps[stepIndex] = { ...state.executionSteps[stepIndex], ...action.payload.updates };
      }
    })
    .addCase(actions.setCurrentStep, (state, action) => {
      state.currentStep = action.payload;
    })
    .addCase(actions.setAnalysisResult, (state, action) => {
      state.analysis = action.payload;
    })
    .addCase(actions.setAnalysisMetadata, (state, action) => {
      state.metadata = action.payload;
    })
    .addCase(actions.setMacroVariables, (state, action) => {
      state.macroVariables = action.payload;
    })
    .addCase(actions.setAssetReturns, (state, action) => {
      state.assetReturns = action.payload;
    })
    .addCase(actions.setOutOfScope, (state, action) => {
      state.isInScope = action.payload.isInScope;
      state.outOfScopeReason = action.payload.reason;
    })
    .addCase(actions.analysisCompleted, (state) => {
      state.isExecuting = false;
      state.lastExecutedAt = new Date();
    })
    .addCase(actions.analysisFailed, (state, action) => {
      state.isExecuting = false;
      state.error = action.payload;
      state.lastExecutedAt = new Date();
    })
    .addCase(actions.resetAnalysis, (state) => {
      state.currentQuery = '';
      state.isExecuting = false;
      state.executionId = null;
      state.executionSteps = [];
      state.currentStep = null;
      state.analysis = null;
      state.error = null;
      state.isInScope = true;
      state.outOfScopeReason = undefined;
      state.macroVariables = [];
      state.assetReturns = [];
    })
    .addCase(actions.clearError, (state) => {
      state.error = null;
    });
});
