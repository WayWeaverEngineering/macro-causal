import { createAction } from '@reduxjs/toolkit';

// UI Actions
export const setLoading = createAction<boolean>('ui/setLoading');
export const setLoadingWithMessage = createAction<{ isLoading: boolean; message?: string }>('ui/setLoadingWithMessage');
export const showProgressBar = createAction<number>('ui/showProgressBar');
export const hideProgressBar = createAction('ui/hideProgressBar');
export const updateProgressBar = createAction<{ progress: number; message: string }>('ui/updateProgressBar');
export const setActiveTab = createAction<'analysis' | 'regime' | 'uncertainty'>('ui/setActiveTab');
export const setOutputView = createAction<'summary' | 'detailed' | 'regime' | 'uncertainty'>('ui/setOutputView');
export const toggleSidebar = createAction('ui/toggleSidebar');
export const setSidebarOpen = createAction<boolean>('ui/setSidebarOpen');
export const setScreenSize = createAction<'xs' | 'sm' | 'md' | 'lg' | 'xl'>('ui/setScreenSize');
export const setTheme = createAction<'light' | 'dark'>('ui/setTheme');
export const setFontSize = createAction<'small' | 'medium' | 'large'>('ui/setFontSize');
export const toggleConfidence = createAction('ui/toggleConfidence');
export const toggleLimitations = createAction('ui/toggleLimitations');
export const showModal = createAction<{ type: 'error' | 'info' | 'warning' | 'success'; title: string; message: string; actions?: Array<{ label: string; action: () => void; variant: 'text' | 'outlined' | 'contained' }> }>('ui/showModal');
export const hideModal = createAction('ui/hideModal');
export const showError = createAction<string>('ui/showError');
export const clearError = createAction('ui/clearError');
