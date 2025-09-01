import { createReducer } from '@reduxjs/toolkit';
import { UserState } from '../states/types';
import { initialUserState } from '../states/initialStates';
import * as actions from '../actions/userActions';

export const userReducer = createReducer<UserState>(initialUserState, (builder) => {
  builder
    .addCase(actions.setUserPreferences, (state, action) => {
      state.preferences = action.payload;
    })
    .addCase(actions.updateUserPreference, (state, action) => {
      state.preferences = {
        ...state.preferences,
        [action.payload.key]: action.payload.value,
      };
    })
    .addCase(actions.addQueryToHistory, (state, action) => {
      state.queryHistory.unshift(action.payload);
      // Keep only last 50 queries
      if (state.queryHistory.length > 50) {
        state.queryHistory = state.queryHistory.slice(0, 50);
      }
    })
    .addCase(actions.clearQueryHistory, (state) => {
      state.queryHistory = [];
    })
    .addCase(actions.saveAnalysis, (state, action) => {
      state.savedAnalyses.push(action.payload);
    })
    .addCase(actions.removeSavedAnalysis, (state, action) => {
      state.savedAnalyses = state.savedAnalyses.filter(analysis => analysis.id !== action.payload);
    })
    .addCase(actions.updateSavedAnalysis, (state, action) => {
      const index = state.savedAnalyses.findIndex(analysis => analysis.id === action.payload.id);
      if (index !== -1) {
        state.savedAnalyses[index] = { ...state.savedAnalyses[index], ...action.payload.updates };
      }
    })
    .addCase(actions.setSessionId, (state, action) => {
      state.sessionId = action.payload;
    })
    .addCase(actions.updateLastActivity, (state) => {
      state.lastActivity = new Date();
    })
    .addCase(actions.addRecentQuery, (state, action) => {
      // Remove if already exists
      state.recentQueries = state.recentQueries.filter(query => query !== action.payload);
      // Add to beginning
      state.recentQueries.unshift(action.payload);
      // Keep only last 10 queries
      if (state.recentQueries.length > 10) {
        state.recentQueries = state.recentQueries.slice(0, 10);
      }
    })
    .addCase(actions.clearRecentQueries, (state) => {
      state.recentQueries = [];
    });
});
