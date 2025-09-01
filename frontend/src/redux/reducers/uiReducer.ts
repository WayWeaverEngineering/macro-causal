import { createReducer } from '@reduxjs/toolkit';
import { UIState } from '../states/types';
import { initialUIState } from '../states/initialStates';
import * as actions from '../actions/uiActions';

export const uiReducer = createReducer<UIState>(initialUIState, (builder) => {
  builder
    .addCase(actions.setLoading, (state, action) => {
      state.isLoading = action.payload;
    })
    .addCase(actions.setLoadingWithMessage, (state, action) => {
      state.isLoading = action.payload.isLoading;
      if (action.payload.message) {
        state.loadingMessage = action.payload.message;
      }
    })
    .addCase(actions.showProgressBar, (state, action) => {
      state.progressBar.isVisible = true;
      state.progressBar.progress = action.payload;
    })
    .addCase(actions.hideProgressBar, (state) => {
      state.progressBar.isVisible = false;
      state.progressBar.progress = 0;
      state.progressBar.message = '';
    })
    .addCase(actions.updateProgressBar, (state, action) => {
      state.progressBar.progress = action.payload.progress;
      state.progressBar.message = action.payload.message;
    })
    .addCase(actions.setActiveTab, (state, action) => {
      state.activeTab = action.payload;
    })
    .addCase(actions.setOutputView, (state, action) => {
      state.outputView = action.payload;
    })
    .addCase(actions.toggleSidebar, (state) => {
      state.sidebarOpen = !state.sidebarOpen;
    })
    .addCase(actions.setSidebarOpen, (state, action) => {
      state.sidebarOpen = action.payload;
    })
    .addCase(actions.setScreenSize, (state, action) => {
      state.screenSize = action.payload;
      state.isMobile = action.payload === 'xs' || action.payload === 'sm';
    })
    .addCase(actions.setTheme, (state, action) => {
      state.theme = action.payload;
    })
    .addCase(actions.setFontSize, (state, action) => {
      state.fontSize = action.payload;
    })
    .addCase(actions.toggleConfidence, (state) => {
      state.showConfidence = !state.showConfidence;
    })
    .addCase(actions.toggleLimitations, (state) => {
      state.showLimitations = !state.showLimitations;
    })
    .addCase(actions.showModal, (state, action) => {
      state.modals = {
        isOpen: true,
        type: action.payload.type,
        title: action.payload.title,
        message: action.payload.message,
        actions: action.payload.actions || [],
      };
    })
    .addCase(actions.hideModal, (state) => {
      state.modals.isOpen = false;
    })
    .addCase(actions.showError, (state, action) => {
      state.modals = {
        isOpen: true,
        type: 'error',
        title: 'Error',
        message: action.payload,
        actions: [
          {
            label: 'Close',
            action: () => {},
            variant: 'text',
          },
        ],
      };
    })
    .addCase(actions.clearError, (state) => {
      state.modals.isOpen = false;
    });
});
