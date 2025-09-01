import { combineReducers } from '@reduxjs/toolkit';
import { analysisReducer } from './analysisReducer';
import { uiReducer } from './uiReducer';
import { userReducer } from './userReducer';

export const rootReducer = combineReducers({
  analysis: analysisReducer,
  ui: uiReducer,
  user: userReducer,
});

export type RootState = ReturnType<typeof rootReducer>;
