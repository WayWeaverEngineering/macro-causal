import { ExecutionStep } from "./AgentModels";

export interface AnalysisRequest {
  query: string;
  sessionId?: string;
}

export interface AnalysisScheduleResponse {
  success: boolean;
  message: string;
  executionId?: string;
  error?: string;
}

export interface AnalysisStatusResponse {
  success: boolean;
  message: string;
  executionId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  userQuery: string;
  sessionId?: string;
  steps: ExecutionStep[];
  currentStep: ExecutionStep | null;
  result?: any;
  error?: string;
  createdAt: string;
  updatedAt: string;
}

export interface AnalysisMessage {
  executionId: string;
  query: string;
  sessionId?: string;
}

export interface AnalysisExecution {
  executionId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  userQuery: string;
  sessionId?: string;
  steps: ExecutionStep[];
  currentStep: ExecutionStep | null;
  result: any;
  error: string | null;
  createdAt: string;
  updatedAt: string;
}
