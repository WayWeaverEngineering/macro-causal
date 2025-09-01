import { RootState } from './store';

// Analysis Selectors
export const selectCurrentQuery = (state: RootState) => state.analysis.currentQuery;
export const selectIsExecuting = (state: RootState) => state.analysis.isExecuting;
export const selectExecutionId = (state: RootState) => state.analysis.executionId;
export const selectExecutionSteps = (state: RootState) => state.analysis.executionSteps;
export const selectCurrentStep = (state: RootState) => state.analysis.currentStep;
export const selectCurrentAnalysis = (state: RootState) => state.analysis.analysis;
export const selectIsInScope = (state: RootState) => state.analysis.isInScope;
export const selectOutOfScopeReason = (state: RootState) => state.analysis.outOfScopeReason;
export const selectAnalysisMetadata = (state: RootState) => state.analysis.metadata;
export const selectMacroVariables = (state: RootState) => state.analysis.macroVariables;
export const selectAssetReturns = (state: RootState) => state.analysis.assetReturns;
export const selectAnalysisError = (state: RootState) => state.analysis.error;
export const selectLastExecutedAt = (state: RootState) => state.analysis.lastExecutedAt;

// UI Selectors
export const selectIsLoading = (state: RootState) => state.ui.isLoading;
export const selectLoadingMessage = (state: RootState) => state.ui.loadingMessage;
export const selectProgressBar = (state: RootState) => state.ui.progressBar;
export const selectActiveTab = (state: RootState) => state.ui.activeTab;
export const selectOutputView = (state: RootState) => state.ui.outputView;
export const selectSidebarOpen = (state: RootState) => state.ui.sidebarOpen;
export const selectScreenSize = (state: RootState) => state.ui.screenSize;
export const selectIsMobile = (state: RootState) => state.ui.isMobile;
export const selectTheme = (state: RootState) => state.ui.theme;
export const selectFontSize = (state: RootState) => state.ui.fontSize;
export const selectShowConfidence = (state: RootState) => state.ui.showConfidence;
export const selectShowLimitations = (state: RootState) => state.ui.showLimitations;
export const selectModals = (state: RootState) => state.ui.modals;

// User Selectors
export const selectUserPreferences = (state: RootState) => state.user.preferences;
export const selectQueryHistory = (state: RootState) => state.user.queryHistory;
export const selectRecentQueries = (state: RootState) => state.user.recentQueries;
export const selectSavedAnalyses = (state: RootState) => state.user.savedAnalyses;
export const selectSessionId = (state: RootState) => state.user.sessionId;
export const selectLastActivity = (state: RootState) => state.user.lastActivity;

// Computed Selectors
export const selectProgressPercentage = (state: RootState) => {
  const steps = state.analysis.executionSteps;
  if (steps.length === 0) return 0;
  const completedSteps = steps.filter(step => step.status === 'completed').length;
  return Math.round((completedSteps / steps.length) * 100);
};

export const selectHasResults = (state: RootState) => {
  return state.analysis.analysis !== null && state.analysis.isInScope;
};

export const selectIsOutOfScope = (state: RootState) => {
  return !state.analysis.isInScope;
};

export const selectAnalysisStatus = (state: RootState) => {
  if (state.analysis.error) return 'failed';
  if (state.analysis.analysis) return 'completed';
  if (state.analysis.isExecuting) return 'running';
  return 'idle';
};
