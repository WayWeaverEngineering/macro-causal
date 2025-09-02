import { createAsyncThunk } from '@reduxjs/toolkit';
import { 
  submitAnalysisRequest, 
  submitAnalysisWithProgress,
  getAnalysisStatus,
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
  setAnalysisStatus,
  updateAnalysisProgress,
  setAnalysisError,
  setAnalysisCreatedAt,
  setAnalysisUpdatedAt,
  setUserQuery,
  setSessionId,
} from '../actions/analysisActions';
import { 
  setLoadingWithMessage, 
  showError,
} from '../actions/uiActions';
import { addRecentQuery } from '../actions/userActions';
import { RootState } from '../store';
import { mapBackendResultToCausalAnalysis } from '../utils/mapBackendToFrontend';

/**
 * Thunk to submit an analysis request
 */
export const submitAnalysisThunk = createAsyncThunk(
  'analysis/submitAnalysis',
  async (
    query: string,
    { dispatch, getState }
  ) => {
    try {
      const state = getState() as RootState;
      const sessionId = state.user.sessionId;
      
      // Start analysis
      dispatch(startAnalysis(query));
      dispatch(setSessionId(sessionId));
      dispatch(setUserQuery(query));
      dispatch(setLoadingWithMessage({ isLoading: true, message: 'Submitting analysis request...' }));
      
      // Submit request to backend
      const response = await submitAnalysisRequest(query, { 
        sessionId,
      });
      
      if (!response.success) {
        dispatch(analysisFailed(response.message || 'Analysis request failed'));
        dispatch(setLoadingWithMessage({ isLoading: false }));
        dispatch(showError(response.message || 'Analysis request failed'));
        return response;
      }
      
      if (!response.executionId) {
        const errorMsg = 'No execution ID received from backend';
        dispatch(analysisFailed(errorMsg));
        dispatch(setLoadingWithMessage({ isLoading: false }));
        dispatch(showError(errorMsg));
        return { success: false, message: errorMsg };
      }
      
      // Set execution ID
      dispatch(setExecutionId(response.executionId));
      
      // Add to recent queries
      dispatch(addRecentQuery(query));
      
      // Start polling for status updates
      dispatch(pollAnalysisStatusThunk(response.executionId));
      
      return response;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      dispatch(analysisFailed(errorMessage));
      dispatch(setLoadingWithMessage({ isLoading: false }));
      dispatch(showError(errorMessage));
      throw error;
    }
  }
);

/**
 * Thunk to poll for analysis status updates
 */
export const pollAnalysisStatusThunk = createAsyncThunk(
  'analysis/pollAnalysisStatus',
  async (
    executionId: string,
    { dispatch }
  ) => {
    try {
      const pollInterval = 2000; // 2 seconds
      const maxPollTime = 300000; // 5 minutes
      const startTime = Date.now();
      
      while (Date.now() - startTime < maxPollTime) {
        const response = await getAnalysisStatus(executionId);
        
        if (!response.success) {
          dispatch(analysisFailed(response.message || 'Status check failed'));
          dispatch(setLoadingWithMessage({ isLoading: false }));
          dispatch(showError(response.message || 'Status check failed'));
          return response;
        }
        
        const status = response.status;
        const steps = response.steps || [];
        const currentStep = response.currentStep;
        
        // Update status if available
        if (status) {
          dispatch(setAnalysisStatus(status));
        }
        
        // Update timestamps if available
        if (response.createdAt) {
          dispatch(setAnalysisCreatedAt(response.createdAt));
        }
        if (response.updatedAt) {
          dispatch(setAnalysisUpdatedAt(response.updatedAt));
        }
        
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
        
        dispatch(updateAnalysisProgress({ progress, message: currentStep?.description || 'Processing...' }));
        
        // Check if analysis is complete or failed
        if (status === 'completed') {
          // Set final results
          if (response.result) {
            const mapped = mapBackendResultToCausalAnalysis(response.result);
            if (mapped) {
              dispatch(setAnalysisResult(mapped));
            }
            if (response.result.metadata) {
              dispatch(setAnalysisMetadata(response.result.metadata));
            }
          }
          if (response.macroVariables) {
            dispatch(setMacroVariables(response.macroVariables));
          }
          if (response.assetReturns) {
            dispatch(setAssetReturns(response.assetReturns));
          }
          
          // Check if out of scope
          if (response.error && response.error.includes('out of scope')) {
            dispatch(setOutOfScope({ isInScope: false, reason: response.error }));
          }
          
          dispatch(analysisCompleted());
          dispatch(setLoadingWithMessage({ isLoading: false }));
          
          return response;
        }
        
        if (status === 'failed') {
          const errorMsg = response.error || 'Analysis failed';
          dispatch(analysisFailed(errorMsg));
          dispatch(setAnalysisError(errorMsg));
          dispatch(setLoadingWithMessage({ isLoading: false }));
          dispatch(showError(errorMsg));
          return response;
        }
        
        // Wait before next poll
        await new Promise(resolve => setTimeout(resolve, pollInterval));
      }
      
      const timeoutMsg = 'Analysis polling timed out';
      dispatch(analysisFailed(timeoutMsg));
      dispatch(setAnalysisError(timeoutMsg));
      dispatch(setLoadingWithMessage({ isLoading: false }));
      dispatch(showError(timeoutMsg));
      
      return { success: false, message: timeoutMsg };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      dispatch(analysisFailed(errorMessage));
      dispatch(setAnalysisError(errorMessage));
      dispatch(setLoadingWithMessage({ isLoading: false }));
      dispatch(showError(errorMessage));
      throw error;
    }
  }
);

/**
 * Thunk to submit analysis with progress updates
 */
export const submitAnalysisWithProgressThunk = createAsyncThunk(
  'analysis/submitAnalysisWithProgress',
  async (
    query: string,
    { dispatch, getState }
  ) => {
    try {
      const state = getState() as RootState;
      const sessionId = state.user.sessionId;
      
      // Start analysis
      dispatch(startAnalysis(query));
      dispatch(setSessionId(sessionId));
      dispatch(setUserQuery(query));
      dispatch(setLoadingWithMessage({ isLoading: true, message: 'Starting analysis...' }));
      
      // Submit request with progress updates
      const response = await submitAnalysisWithProgress(
        query,
        (_, progress, message) => {
          dispatch(setLoadingWithMessage({ isLoading: true, message }));
          dispatch(updateAnalysisProgress({ progress, message }));
        },
        { 
          sessionId,
          pollInterval: 2000,
          maxPollTime: 300000,
        }
      );
      
      if (!response.success) {
        dispatch(analysisFailed(response.message || 'Analysis failed'));
        dispatch(setLoadingWithMessage({ isLoading: false }));
        dispatch(showError(response.message || 'Analysis failed'));
        return response;
      }
      
      // Handle the response
      if (response.result) {
        const mapped = mapBackendResultToCausalAnalysis(response.result);
        if (mapped) {
          dispatch(setAnalysisResult(mapped));
        }
        if (response.result.metadata) {
          dispatch(setAnalysisMetadata(response.result.metadata));
        }
      }
      if (response.macroVariables) {
        dispatch(setMacroVariables(response.macroVariables));
      }
      if (response.assetReturns) {
        dispatch(setAssetReturns(response.assetReturns));
      }
      
      // Check if out of scope
      if (response.error && response.error.includes('out of scope')) {
        dispatch(setOutOfScope({ isInScope: false, reason: response.error }));
      }
      
      dispatch(analysisCompleted());
      dispatch(setLoadingWithMessage({ isLoading: false }));
      
      return response;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      dispatch(analysisFailed(errorMessage));
      dispatch(setAnalysisError(errorMessage));
      dispatch(setLoadingWithMessage({ isLoading: false }));
      dispatch(showError(errorMessage));
      throw error;
    }
  }
);

// Legacy thunks for backward compatibility
export const submitCausalAnalysisThunk = submitAnalysisThunk;
export const pollCausalAnalysisStatusThunk = pollAnalysisStatusThunk;
export const submitCausalAnalysisWithProgressThunk = submitAnalysisWithProgressThunk;
