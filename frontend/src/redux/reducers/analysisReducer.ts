import { createReducer } from '@reduxjs/toolkit';
import { AnalysisState } from '../states/types';
import { initialAnalysisState } from '../states/initialStates';
import * as actions from '../actions/analysisActions';

export const analysisReducer = createReducer<AnalysisState>(initialAnalysisState, (builder) => {
  builder
    .addCase(actions.startAnalysis, (state, action) => {
      state.currentQuery = action.payload;
      state.userQuery = action.payload;
      state.isExecuting = true;
      state.executionId = null;
      state.executionSteps = [];
      state.currentStep = null;
      state.analysis = null;
      state.error = null;
      state.isInScope = true;
      state.outOfScopeReason = undefined;
      state.status = 'pending';
      state.progress = 0;
      state.progressMessage = 'Starting analysis...';
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
      state.status = 'completed';
      state.progress = 100;
      state.progressMessage = 'Analysis completed';
      state.lastExecutedAt = new Date();
    })
    .addCase(actions.analysisFailed, (state, action) => {
      state.isExecuting = false;
      state.status = 'failed';
      state.error = action.payload;
      state.progressMessage = 'Analysis failed';
      state.lastExecutedAt = new Date();
    })
    .addCase(actions.resetAnalysis, (state) => {
      state.currentQuery = '';
      state.userQuery = '';
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
      state.status = 'pending';
      state.progress = 0;
      state.progressMessage = '';
      state.createdAt = null;
      state.updatedAt = null;
    })
    .addCase(actions.clearError, (state) => {
      state.error = null;
    })
    // New cases for status updates
    .addCase(actions.setAnalysisStatus, (state, action) => {
      state.status = action.payload;
    })
    .addCase(actions.updateAnalysisProgress, (state, action) => {
      state.progress = action.payload.progress;
      state.progressMessage = action.payload.message;
    })
    .addCase(actions.setAnalysisError, (state, action) => {
      state.error = action.payload;
    })
    .addCase(actions.setAnalysisCreatedAt, (state, action) => {
      state.createdAt = action.payload;
    })
    .addCase(actions.setAnalysisUpdatedAt, (state, action) => {
      state.updatedAt = action.payload;
    })
    .addCase(actions.setUserQuery, (state, action) => {
      state.userQuery = action.payload;
    })
    .addCase(actions.setSessionId, (state, action) => {
      state.sessionId = action.payload;
    });
});
