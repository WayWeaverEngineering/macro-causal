import { createAction } from '@reduxjs/toolkit';
import { UserPreferences, QueryHistoryItem, SavedAnalysis } from '../states/types';

// User Actions
export const setUserPreferences = createAction<UserPreferences>('user/setUserPreferences');
export const updateUserPreference = createAction<{ key: keyof UserPreferences; value: any }>('user/updateUserPreference');
export const addQueryToHistory = createAction<QueryHistoryItem>('user/addQueryToHistory');
export const clearQueryHistory = createAction('user/clearQueryHistory');
export const saveAnalysis = createAction<SavedAnalysis>('user/saveAnalysis');
export const removeSavedAnalysis = createAction<string>('user/removeSavedAnalysis');
export const updateSavedAnalysis = createAction<{ id: string; updates: Partial<SavedAnalysis> }>('user/updateSavedAnalysis');
export const setSessionId = createAction<string>('user/setSessionId');
export const updateLastActivity = createAction('user/updateLastActivity');
export const addRecentQuery = createAction<string>('user/addRecentQuery');
export const clearRecentQueries = createAction('user/clearRecentQueries');
