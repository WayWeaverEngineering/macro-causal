import { AnalysisState, UIState, UserState } from './types';

export const initialAnalysisState: AnalysisState = {
  currentQuery: '',
  isExecuting: false,
  executionId: null,
  executionSteps: [],
  currentStep: null,
  analysis: null,
  isInScope: true,
  outOfScopeReason: undefined,
  metadata: {
    executionTime: 0,
    dataPointsAnalyzed: 0,
    timePeriod: '',
    confidence: 0,
    modelVersion: '1.0.0'
  },
  macroVariables: [],
  assetReturns: [],
  error: null,
  lastExecutedAt: null,
};

export const initialUIState: UIState = {
  sidebarOpen: false,
  activeTab: 'analysis',
  outputView: 'summary',
  isLoading: false,
  loadingMessage: '',
  progressBar: {
    isVisible: false,
    progress: 0,
    message: '',
  },
  modals: {
    isOpen: false,
    type: 'info',
    title: '',
    message: '',
    actions: [],
  },
  isMobile: false,
  screenSize: 'lg',
  theme: 'dark',
  fontSize: 'medium',
  showConfidence: true,
  showLimitations: true,
};

export const initialUserState: UserState = {
  preferences: {
    defaultTimeframe: '1Y',
    preferredAssets: ['SP500', 'TLT', 'GLD', 'USO', 'VIX'],
    confidenceThreshold: 0.8,
    showTechnicalDetails: false,
    autoRefresh: false,
  },
  queryHistory: [],
  recentQueries: [
    "What's the causal effect of a 1% Fed rate hike on S&P 500 returns?",
    "How do CPI surprises affect bond returns?",
    "What market regime are we currently in and how does it affect causal relationships?",
    "Analyze the causal relationship between GDP growth and equity returns",
    "How do oil price shocks impact inflation and asset returns?"
  ],
  savedAnalyses: [],
  sessionId: `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
  lastActivity: new Date(),
};
