import { createAsyncThunk } from '@reduxjs/toolkit';
import { 
  submitCausalAnalysisRequest, 
  submitCausalAnalysisWithProgress,
  getCausalAnalysisStatus,
} from '../../app/api/dataBridge';
import { 
  startAnalysis,
  analysisFailed,
  setExecutionId,
  addExecutionStep,
  updateExecutionStep,
  setCurrentStep,
  setAnalysisResult,
  setAnalysisMetadata,
  setMacroVariables,
  setAssetReturns,
  setOutOfScope,
  analysisCompleted,
} from '../actions/analysisActions';
import { 
  setLoadingWithMessage, 
  showProgressBar, 
  hideProgressBar,
  showError,
  updateProgressBar,
} from '../actions/uiActions';
import { addRecentQuery } from '../actions/userActions';
import { RootState } from '../store';

/**
 * Thunk to submit a causal analysis request
 */
export const submitCausalAnalysisThunk = createAsyncThunk(
  'analysis/submitCausalAnalysis',
  async (
    query: string,
    { dispatch, getState }
  ) => {
    try {
      const state = getState() as RootState;
      const sessionId = state.user.sessionId;
      const preferences = state.user.preferences;
      
      // Start analysis
      dispatch(startAnalysis(query));
      dispatch(setLoadingWithMessage({ isLoading: true, message: 'Submitting causal analysis request...' }));
      dispatch(showProgressBar(0));
      
      // Submit request to backend
      const response = await submitCausalAnalysisRequest(query, { 
        sessionId,
        macroVariables: preferences.preferredAssets,
        assets: preferences.preferredAssets,
        timeframe: preferences.defaultTimeframe,
      });
      
      if (!response.success) {
        dispatch(analysisFailed(response.message));
        dispatch(setLoadingWithMessage({ isLoading: false }));
        dispatch(hideProgressBar());
        dispatch(showError(response.message));
        return response;
      }
      
      if (!response.executionId) {
        const errorMsg = 'No execution ID received from backend';
        dispatch(analysisFailed(errorMsg));
        dispatch(setLoadingWithMessage({ isLoading: false }));
        dispatch(hideProgressBar());
        dispatch(showError(errorMsg));
        return { success: false, message: errorMsg };
      }
      
      // Set execution ID
      dispatch(setExecutionId(response.executionId));
      
      // Add to recent queries
      dispatch(addRecentQuery(query));
      
      // Start polling for status updates
      dispatch(pollCausalAnalysisStatusThunk(response.executionId));
      
      return response;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      dispatch(analysisFailed(errorMessage));
      dispatch(setLoadingWithMessage({ isLoading: false }));
      dispatch(hideProgressBar());
      dispatch(showError(errorMessage));
      throw error;
    }
  }
);

/**
 * Thunk to poll for causal analysis status updates
 */
export const pollCausalAnalysisStatusThunk = createAsyncThunk(
  'analysis/pollCausalAnalysisStatus',
  async (
    executionId: string,
    { dispatch }
  ) => {
    try {
      const pollInterval = 2000; // 2 seconds
      const maxPollTime = 300000; // 5 minutes
      const startTime = Date.now();
      
      while (Date.now() - startTime < maxPollTime) {
        const response = await getCausalAnalysisStatus(executionId);
        
        if (!response.success) {
          dispatch(analysisFailed(response.message));
          dispatch(setLoadingWithMessage({ isLoading: false }));
          dispatch(hideProgressBar());
          dispatch(showError(response.message));
          return response;
        }
        
        const status = response.status;
        const steps = response.steps || [];
        const currentStep = response.currentStep;
        
        // Update execution steps
        steps.forEach(step => {
          dispatch(addExecutionStep(step));
          if (step.status === 'completed') {
            dispatch(updateExecutionStep({ stepId: step.stepId, updates: { status: 'completed' } }));
          }
        });
        
        // Update current step
        if (currentStep) {
          dispatch(setCurrentStep(currentStep));
        }
        
        // Update progress
        const completedSteps = steps.filter(step => step.status === 'completed').length;
        const totalSteps = steps.length;
        const progress = totalSteps > 0 ? Math.round((completedSteps / totalSteps) * 100) : 0;
        
        dispatch(updateProgressBar({ progress, message: currentStep?.description || 'Processing...' }));
        
        // Check if analysis is complete or failed
        if (status === 'completed') {
          // Set final results
          if (response.analysis) {
            dispatch(setAnalysisResult(response.analysis));
          }
          if (response.metadata) {
            dispatch(setAnalysisMetadata(response.metadata));
          }
          if (response.macroVariables) {
            dispatch(setMacroVariables(response.macroVariables));
          }
          if (response.assetReturns) {
            dispatch(setAssetReturns(response.assetReturns));
          }
          
          // Check if out of scope
          if (response.outOfScopeReason) {
            dispatch(setOutOfScope({ isInScope: false, reason: response.outOfScopeReason }));
          }
          
          dispatch(analysisCompleted());
          dispatch(setLoadingWithMessage({ isLoading: false }));
          dispatch(hideProgressBar());
          
          return response;
        }
        
        if (status === 'failed') {
          const errorMsg = response.error || 'Analysis failed';
          dispatch(analysisFailed(errorMsg));
          dispatch(setLoadingWithMessage({ isLoading: false }));
          dispatch(hideProgressBar());
          dispatch(showError(errorMsg));
          return response;
        }
        
        // Wait before next poll
        await new Promise(resolve => setTimeout(resolve, pollInterval));
      }
      
      const timeoutMsg = 'Analysis polling timed out';
      dispatch(analysisFailed(timeoutMsg));
      dispatch(setLoadingWithMessage({ isLoading: false }));
      dispatch(hideProgressBar());
      dispatch(showError(timeoutMsg));
      
      return { success: false, message: timeoutMsg };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      dispatch(analysisFailed(errorMessage));
      dispatch(setLoadingWithMessage({ isLoading: false }));
      dispatch(hideProgressBar());
      dispatch(showError(errorMessage));
      throw error;
    }
  }
);

/**
 * Thunk to submit analysis with progress updates
 */
export const submitCausalAnalysisWithProgressThunk = createAsyncThunk(
  'analysis/submitCausalAnalysisWithProgress',
  async (
    query: string,
    { dispatch, getState }
  ) => {
    try {
      const state = getState() as RootState;
      const sessionId = state.user.sessionId;
      const preferences = state.user.preferences;
      
      // Start analysis
      dispatch(startAnalysis(query));
      dispatch(setLoadingWithMessage({ isLoading: true, message: 'Starting causal analysis...' }));
      dispatch(showProgressBar(0));
      
      // Submit request with progress updates
      const response = await submitCausalAnalysisWithProgress(
        query,
        (_, progress, message) => {
          dispatch(setLoadingWithMessage({ isLoading: true, message }));
          dispatch(updateProgressBar({ progress, message }));
        },
        { 
          sessionId,
          macroVariables: preferences.preferredAssets,
          assets: preferences.preferredAssets,
          timeframe: preferences.defaultTimeframe,
          pollInterval: 2000,
          maxPollTime: 300000,
        }
      );
      
      if (!response.success) {
        dispatch(analysisFailed(response.message));
        dispatch(setLoadingWithMessage({ isLoading: false }));
        dispatch(hideProgressBar());
        dispatch(showError(response.message));
        return response;
      }
      
      // Handle the response
      if (response.analysis) {
        dispatch(setAnalysisResult(response.analysis));
      }
      if (response.metadata) {
        dispatch(setAnalysisMetadata(response.metadata));
      }
      if (response.macroVariables) {
        dispatch(setMacroVariables(response.macroVariables));
      }
      if (response.assetReturns) {
        dispatch(setAssetReturns(response.assetReturns));
      }
      
      // Check if out of scope
      if (response.outOfScopeReason) {
        dispatch(setOutOfScope({ isInScope: false, reason: response.outOfScopeReason }));
      }
      
      dispatch(analysisCompleted());
      dispatch(setLoadingWithMessage({ isLoading: false }));
      dispatch(hideProgressBar());
      
      return response;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      dispatch(analysisFailed(errorMessage));
      dispatch(setLoadingWithMessage({ isLoading: false }));
      dispatch(hideProgressBar());
      dispatch(showError(errorMessage));
      throw error;
    }
  }
);
