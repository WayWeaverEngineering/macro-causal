# Macro AI Analyst - Frontend

## Overview

The frontend is a React-based web application that provides an intuitive interface for AI-powered macroeconomic causal analysis. It uses Redux Toolkit for state management and integrates with a sophisticated backend system that processes causal analysis requests using X-Learner and Regime Classifier models.

## Features

### 🎯 **Query Input**
- **Multi-line text input** for complex macroeconomic questions
- **Enter key submission** for quick analysis initiation
- **Submit button** with loading states
- **Example queries** for inspiration and quick testing
- **Real-time validation** and error handling

### 📊 **Real-time Analysis Status**
- **Progress bar** showing completion percentage
- **Current step indicator** with detailed descriptions
- **Execution step tracking** with status icons
- **Error handling** with clear error messages
- **Collapsible interface** that appears only when needed

### 📈 **Comprehensive Causal Analysis Results**
- **Causal Effect Display** with confidence intervals and p-values
- **Market Regime Analysis** showing different market states
- **Uncertainty Estimation** with reliability indicators
- **Key Insights** with bullet-point formatting
- **Methodology** explanation of the causal inference approach
- **Limitations** and analysis constraints

### 🎨 **Modern UI Design**
- **Dark theme** optimized for readability
- **Material-UI components** for consistent design
- **Responsive layout** that works on all devices
- **Smooth animations** and transitions
- **Accessibility features** for inclusive design

## Quick Start

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Backend API running (see backend README)

### Installation
```bash
npm install
```

### Development
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

### Production Build
```bash
npm run build
```

## Usage Guide

### 1. **Ask a Macro Question**
- Type your analysis question in the text area
- Use specific macroeconomic variables, monetary policy, or asset classes
- Examples:
  - "What's the causal effect of a 1% Fed rate hike on S&P 500 returns?"
  - "How do CPI surprises affect bond returns?"
  - "What market regime are we currently in and how does it affect causal relationships?"

### 2. **Submit Analysis**
- Press **Enter** or click the **"Analyze Causally"** button
- The system will immediately show progress indicators
- Real-time updates will display current processing steps

### 3. **Monitor Progress**
- Watch the **progress bar** for completion percentage
- View **current step** details and descriptions
- See **execution history** with status indicators

### 4. **Review Results**
- **Causal Effect**: True causal relationship with confidence intervals
- **Market Regime**: Current market state and regime probabilities
- **Uncertainty**: Analysis reliability and uncertainty factors
- **Key Insights**: Main findings and observations
- **Methodology**: Explanation of the causal inference approach
- **Limitations**: Analysis constraints and caveats

## Example Queries

### Monetary Policy Analysis
```
"What's the causal effect of a 1% Fed rate hike on S&P 500 returns?"
"How do Fed rate changes impact bond yields?"
"Analyze the causal relationship between monetary policy and inflation"
```

### Economic Indicators
```
"How do CPI surprises affect bond returns?"
"Analyze the causal relationship between GDP growth and equity returns"
"How do unemployment changes impact market volatility?"
```

### Asset Class Relationships
```
"How do oil price shocks impact inflation and asset returns?"
"Analyze the causal relationship between VIX and equity returns"
"How do currency movements affect commodity prices?"
```

### Regime Analysis
```
"What market regime are we currently in and how does it affect causal relationships?"
"Analyze regime-dependent effects of monetary policy"
"How do causal relationships change across different market conditions?"
```

## Architecture

### Component Structure
```
MacroAnalyst (Main Layout)
├── Header (App title and description)
├── QueryInput (Question input and submission)
├── AnalysisStatus (Progress tracking and status)
└── CausalAnalysisResults (Results display)
```

### State Management
- **Redux Toolkit** for global state management
- **Async thunks** for API communication
- **Real-time polling** for status updates
- **Error handling** with user-friendly messages

### API Integration
- **RESTful endpoints** for causal analysis submission
- **Polling mechanism** for status updates
- **Error recovery** and retry logic
- **Timeout handling** for long-running analyses

## Technical Details

### Key Technologies
- **React 18** with functional components and hooks
- **Material-UI** for component library and theming
- **Redux Toolkit** for state management
- **TypeScript** for type safety
- **Vite** for fast development and building

### State Structure
```typescript
interface RootState {
  analysis: AnalysisState;    // Causal analysis execution and results
  ui: UIState;               // UI state and loading indicators
  user: UserState;           // User preferences and history
}
```

### API Endpoints
- `POST /causal-analysis` - Submit causal analysis request
- `GET /causal-analysis/{executionId}` - Get status and results

## Development

### Project Structure
```
src/
├── app/
│   ├── components/          # React components
│   ├── layout/             # Layout components
│   └── style/              # Global styles
├── redux/                  # State management
│   ├── actions/            # Redux actions
│   ├── reducers/           # State reducers
│   ├── selectors/          # State selectors
│   ├── thunks/             # Async thunks
│   └── store/              # Store configuration
└── models/                 # TypeScript interfaces
```

### Available Scripts
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run lint` - Run ESLint
- `npm run preview` - Preview production build

### Environment Variables
Create a `.env` file in the frontend directory:
```env
VITE_API_BASE_URL=http://localhost:8000
```

## Troubleshooting

### Common Issues

**Build Errors**
- Ensure all dependencies are installed: `npm install`
- Check TypeScript errors: `npm run build`
- Verify API endpoint configuration

**Runtime Errors**
- Check browser console for error messages
- Verify backend API is running and accessible
- Ensure CORS is properly configured on backend

**Performance Issues**
- Analysis polling occurs every 2 seconds
- Maximum polling time is 5 minutes
- Large result sets may take time to render

### Support
For issues related to:
- **Frontend UI**: Check component logs and Redux DevTools
- **API Integration**: Verify network requests and responses
- **Backend Processing**: Check backend logs and model outputs

## Future Enhancements

### Planned Features
1. **WebSocket Integration** - Real-time updates instead of polling
2. **Offline Support** - Cache analysis results locally
3. **Batch Analysis** - Multiple concurrent analyses
4. **Advanced Filtering** - Enhanced macro variable and asset filters
5. **Export Functionality** - PDF/Excel export of results
6. **User Authentication** - Personalized analysis history

### Performance Improvements
1. **Virtual Scrolling** - For large result sets
2. **Lazy Loading** - For detailed analysis components
3. **Service Workers** - For offline functionality
4. **Optimistic Updates** - For better perceived performance

The frontend provides a robust, user-friendly interface for macroeconomic causal analysis with comprehensive error handling, real-time progress tracking, and detailed result presentation using state-of-the-art causal inference models.
