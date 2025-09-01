import { createAction } from '@reduxjs/toolkit';
import { 
  ExecutionStep, 
  CausalAnalysisResult, 
  AnalysisMetadata,
  MacroVariable,
  AssetReturn
} from '../../models/analysis';

// Analysis Actions
export const startAnalysis = createAction<string>('analysis/startAnalysis');
export const setExecutionId = createAction<string>('analysis/setExecutionId');
export const addExecutionStep = createAction<ExecutionStep>('analysis/addExecutionStep');
export const updateExecutionStep = createAction<{ stepId: string; updates: Partial<ExecutionStep> }>('analysis/updateExecutionStep');
export const setCurrentStep = createAction<ExecutionStep | null>('analysis/setCurrentStep');
export const setAnalysisResult = createAction<CausalAnalysisResult>('analysis/setAnalysisResult');
export const setAnalysisMetadata = createAction<AnalysisMetadata>('analysis/setAnalysisMetadata');
export const setMacroVariables = createAction<MacroVariable[]>('analysis/setMacroVariables');
export const setAssetReturns = createAction<AssetReturn[]>('analysis/setAssetReturns');
export const setOutOfScope = createAction<{ isInScope: boolean; reason?: string }>('analysis/setOutOfScope');
export const analysisCompleted = createAction('analysis/analysisCompleted');
export const analysisFailed = createAction<string>('analysis/analysisFailed');
export const resetAnalysis = createAction('analysis/resetAnalysis');
export const clearError = createAction('analysis/clearError');
