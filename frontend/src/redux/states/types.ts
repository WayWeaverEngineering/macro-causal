// Import models from the models folder
import {
  ExecutionStep,
  CausalAnalysisResult,
  AnalysisMetadata,
  BackendAnalysisResponse,
  MacroVariable,
  AssetReturn
} from '../../models/analysis';

// Root State Types
export interface RootState {
  analysis: AnalysisState;
  ui: UIState;
  user: UserState;
}

// Analysis State Types
export interface AnalysisState {
  currentQuery: string;
  isExecuting: boolean;
  executionId: string | null;
  executionSteps: ExecutionStep[];
  currentStep: ExecutionStep | null;
  analysis: CausalAnalysisResult | null;
  isInScope: boolean;
  outOfScopeReason?: string;
  metadata: AnalysisMetadata;
  macroVariables: MacroVariable[];
  assetReturns: AssetReturn[];
  error: string | null;
  lastExecutedAt: Date | null;
  // New fields from backend API
  status: 'pending' | 'running' | 'completed' | 'failed';
  userQuery: string;
  sessionId: string | null;
  createdAt: string | null;
  updatedAt: string | null;
  progress: number;
  progressMessage: string;
}

// UI State Types
export interface UIState {
  sidebarOpen: boolean;
  activeTab: 'analysis' | 'regime' | 'uncertainty';
  outputView: 'summary' | 'detailed' | 'regime' | 'uncertainty';
  isLoading: boolean;
  loadingMessage: string;
  progressBar: ProgressBarState;
  modals: ModalState;
  isMobile: boolean;
  screenSize: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  theme: 'light' | 'dark';
  fontSize: 'small' | 'medium' | 'large';
  showConfidence: boolean;
  showLimitations: boolean;
}

// User State Types
export interface UserState {
  preferences: UserPreferences;
  queryHistory: QueryHistoryItem[];
  recentQueries: string[];
  savedAnalyses: SavedAnalysis[];
  sessionId: string;
  lastActivity: Date;
}

// Progress Bar State
export interface ProgressBarState {
  isVisible: boolean;
  progress: number;
  message: string;
}

// Modal State
export interface ModalState {
  isOpen: boolean;
  type: 'error' | 'info' | 'warning' | 'success';
  title: string;
  message: string;
  actions?: Array<{
    label: string;
    action: () => void;
    variant: 'text' | 'outlined' | 'contained';
  }>;
}

// User Preferences
export interface UserPreferences {
  defaultTimeframe: string;
  preferredAssets: string[];
  confidenceThreshold: number;
  showTechnicalDetails: boolean;
  autoRefresh: boolean;
}

// Query History Item
export interface QueryHistoryItem {
  id: string;
  query: string;
  timestamp: Date;
  result: CausalAnalysisResult | null;
  executionTime: number;
}

// Saved Analysis
export interface SavedAnalysis {
  id: string;
  name: string;
  query: string;
  result: CausalAnalysisResult;
  savedAt: Date;
  tags: string[];
}

// Re-export commonly used types for convenience
export type {
  ExecutionStep,
  CausalAnalysisResult,
  AnalysisMetadata,
  BackendAnalysisResponse,
  MacroVariable,
  AssetReturn
};
