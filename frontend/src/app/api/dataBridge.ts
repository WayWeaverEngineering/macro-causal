import axios, { AxiosResponse } from 'axios';
import { ApiGatewayPaths } from './pathMapper';
import { BackendAnalysisResponse } from '../../models/analysis';

axios.defaults.baseURL = ApiGatewayPaths.apiGatewayBaseUrl;
axios.defaults.headers.post['Content-Type'] = 'application/json';

const getResponseBody = <T>(response: AxiosResponse<T | null>) => response.data;

const ApiRequests = {
  get: <T>(relativePath: string, params: URLSearchParams | null = null) => axios.get<T>(relativePath, { params }).then(getResponseBody),
  post: <T>(relativePath: string, payload: {}, params: URLSearchParams | null = null) => axios.post<T>(relativePath, payload, { params, timeout: 90000 }).then(getResponseBody),
  put: <T>(relativePath: string, payload: {}, params: URLSearchParams | null = null) => axios.put<T>(relativePath, payload, { params }).then(getResponseBody),
  delete: <T>(relativePath: string, params: URLSearchParams | null = null) => axios.delete<T>(relativePath, { params }).then(getResponseBody)
};

/**
 * Submit a causal analysis request to the backend
 * @param query - The user's analysis query
 * @param options - Optional parameters for the analysis request
 * @returns Promise<BackendAnalysisResponse> - The backend response with execution ID
 */
export const submitCausalAnalysisRequest = async (
  query: string,
  options?: {
    sessionId?: string;
    timeout?: number;
    macroVariables?: string[];
    assets?: string[];
    timeframe?: string;
  }
): Promise<BackendAnalysisResponse> => {
  try {
    const payload = {
      query,
      sessionId: options?.sessionId,
      macroVariables: options?.macroVariables || ['GDP', 'CPI', 'FedRate', 'Unemployment', 'OilPrice'],
      assets: options?.assets || ['SP500', 'TLT', 'GLD', 'USO', 'VIX'],
      timeframe: options?.timeframe || '1Y',
    };

    const response = await ApiRequests.post<BackendAnalysisResponse>(
      ApiGatewayPaths.causalAnalysisPath,
      payload
    );

    return response || {
      success: false,
      message: 'No response received from backend',
    };
  } catch (error) {
    console.error('Causal analysis request failed:', error);
    
    // Handle different types of errors
    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        return {
          success: false,
          message: 'Request timed out. Please try again with a simpler query.',
        };
      }
      
      if (error.response?.status === 429) {
        return {
          success: false,
          message: 'Too many requests. Please wait a moment and try again.',
        };
      }
      
      if (error.response?.status && error.response.status >= 500) {
        return {
          success: false,
          message: 'Server error. Please try again later.',
        };
      }
      
      return {
        success: false,
        message: error.response?.data?.message || error.message || 'Network error occurred',
      };
    }
    
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Unknown error occurred',
    };
  }
};

/**
 * Get the status of a causal analysis execution
 * @param executionId - The execution ID returned from submitCausalAnalysisRequest
 * @returns Promise<BackendAnalysisResponse> - The current status and results
 */
export const getCausalAnalysisStatus = async (
  executionId: string
): Promise<BackendAnalysisResponse> => {
  try {
    const response = await ApiRequests.get<BackendAnalysisResponse>(
      ApiGatewayPaths.getCausalAnalysisStatusPath(executionId)
    );

    return response || {
      success: false,
      message: 'No response received from backend',
    };
  } catch (error) {
    console.error('Causal analysis status request failed:', error);
    
    if (axios.isAxiosError(error)) {
      if (error.response?.status === 404) {
        return {
          success: false,
          message: 'Analysis execution not found',
        };
      }
      
      if (error.response?.status && error.response.status >= 500) {
        return {
          success: false,
          message: 'Server error. Please try again later.',
        };
      }
      
      return {
        success: false,
        message: error.response?.data?.message || error.message || 'Network error occurred',
      };
    }
    
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Unknown error occurred',
    };
  }
};

/**
 * Poll for causal analysis status updates
 * @param executionId - The execution ID to poll
 * @param onProgress - Callback for progress updates
 * @param options - Polling options
 * @returns Promise<BackendAnalysisResponse> - Final result
 */
export const pollCausalAnalysisStatus = async (
  executionId: string,
  onProgress?: (status: string, progress: number, message: string) => void,
  options?: {
    pollInterval?: number;
    maxPollTime?: number;
    timeout?: number;
  }
): Promise<BackendAnalysisResponse> => {
  const pollInterval = options?.pollInterval || 2000; // 2 seconds
  const maxPollTime = options?.maxPollTime || 300000; // 5 minutes
  const startTime = Date.now();
  
  try {
    while (Date.now() - startTime < maxPollTime) {
      const response = await getCausalAnalysisStatus(executionId);
      
      if (!response.success) {
        return response;
      }
      
      const status = response.status ?? "";
      const steps = response.steps || [];
      const currentStep = response.currentStep;
      
      // Calculate progress based on completed steps
      const completedSteps = steps.filter((step: any) => step.status === 'completed').length;
      const totalSteps = steps.length;
      const progress = totalSteps > 0 ? Math.round((completedSteps / totalSteps) * 100) : 0;
      
      // Call progress callback
      onProgress?.(status, progress, currentStep?.description || 'Processing...');
      
      // Check if analysis is complete
      if (status === 'completed') {
        return response;
      }
      
      // Check if analysis failed
      if (status === 'failed') {
        return {
          success: false,
          message: response.error || 'Analysis failed',
        };
      }
      
      // Wait before next poll
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }
    
    return {
      success: false,
      message: 'Analysis polling timed out',
    };
  } catch (error) {
    console.error('Causal analysis polling failed:', error);
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Polling error occurred',
    };
  }
};

/**
 * Submit a causal analysis request with real-time progress updates
 * @param query - The user's analysis query
 * @param onProgress - Callback function for progress updates
 * @param options - Optional parameters for the analysis request
 * @returns Promise<BackendAnalysisResponse> - The backend response
 */
export const submitCausalAnalysisWithProgress = async (
  query: string,
  onProgress?: (status: string, progress: number, message: string) => void,
  options?: {
    sessionId?: string;
    timeout?: number;
    pollInterval?: number;
    maxPollTime?: number;
    macroVariables?: string[];
    assets?: string[];
    timeframe?: string;
  }
): Promise<BackendAnalysisResponse> => {
  try {
    // Submit the analysis request
    onProgress?.('pending', 0, 'Submitting causal analysis request...');
    
    const submitResponse = await submitCausalAnalysisRequest(query, options);
    
    if (!submitResponse.success || !submitResponse.executionId) {
      return submitResponse;
    }
    
    onProgress?.('pending', 0, 'Analysis scheduled, starting processing...');
    
    // Poll for status updates
    const finalResponse = await pollCausalAnalysisStatus(
      submitResponse.executionId,
      onProgress,
      {
        pollInterval: options?.pollInterval || 2000,
        maxPollTime: options?.maxPollTime || 300000,
      }
    );
    
    return finalResponse;
  } catch (error) {
    console.error('Causal analysis request with progress failed:', error);
    onProgress?.('failed', 0, 'Analysis failed');
    
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Unknown error occurred',
    };
  }
};
