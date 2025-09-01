import { useState } from 'react';
import { useSelector } from 'react-redux';
import { useAppDispatch } from '../../redux/store';
import { 
  Box, 
  TextField, 
  Button, 
  Paper, 
  Typography,
  Chip,
  Stack,
  CircularProgress,
  Alert
} from '@mui/material';
import { Send, Lightbulb, Speed } from '@mui/icons-material';
import { submitCausalAnalysisThunk } from '../../redux/thunks/analysisThunks';
import { 
  selectIsExecuting, 
  selectCurrentQuery, 
  selectCurrentStep, 
  selectLoadingMessage 
} from '../../redux/selectors';

const exampleQueries = [
  "What's the causal effect of a 1% Fed rate hike on S&P 500 returns?",
  "How do CPI surprises affect bond returns?",
  "What market regime are we currently in and how does it affect causal relationships?",
  "Analyze the causal relationship between GDP growth and equity returns",
  "How do oil price shocks impact inflation and asset returns?"
];

export const QueryInput = () => {
  const dispatch = useAppDispatch();
  const isExecuting = useSelector(selectIsExecuting);
  const currentQuery = useSelector(selectCurrentQuery);
  const currentStep = useSelector(selectCurrentStep);
  const loadingMessage = useSelector(selectLoadingMessage);
  
  const [query, setQuery] = useState(currentQuery || '');

  const handleSubmit = () => {
    if (query.trim() && !isExecuting) {
      dispatch(submitCausalAnalysisThunk(query.trim()));
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
    }
  };

  const handleExampleClick = (exampleQuery: string) => {
    setQuery(exampleQuery);
  };

  return (
    <Paper sx={{ p: 3, mb: 3 }}>
      <Typography variant="h6" sx={{ mb: 2, color: '#fff' }}>
        Ask Your Macro Question
      </Typography>
      
      <Box sx={{ mb: 3 }}>
        <TextField
          fullWidth
          multiline
          rows={3}
          variant="outlined"
          placeholder={isExecuting ? "Causal analysis in progress..." : "e.g., What's the causal effect of a 1% Fed rate hike on S&P 500 returns?"}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyPress}
          disabled={isExecuting}
          sx={{
            '& .MuiInputBase-input': {
              color: '#fff',
              fontSize: '16px',
            },
            '& .MuiInputBase-input::placeholder': {
              color: isExecuting ? '#90caf9' : '#888',
              opacity: 1,
            },
            '& .MuiOutlinedInput-root': {
              borderColor: isExecuting ? '#90caf9' : '#444',
              '&:hover fieldset': {
                borderColor: isExecuting ? '#90caf9' : '#666',
              },
              '&.Mui-focused fieldset': {
                borderColor: isExecuting ? '#90caf9' : '#90caf9',
              },
            },
          }}
        />
      </Box>

      {/* Current Step Display in Query Input */}
      {isExecuting && currentStep && (
        <Box sx={{ mb: 2 }}>
          <Alert 
            severity="info" 
            icon={<Speed />}
            sx={{ 
              backgroundColor: '#1a3a5f',
              border: '1px solid #90caf9',
              '& .MuiAlert-icon': {
                color: '#90caf9',
              },
              '& .MuiAlert-message': {
                color: '#fff',
              }
            }}
          >
            <Typography variant="body2" sx={{ color: '#fff', fontWeight: 500 }}>
              Current Step: {currentStep.stepName}
            </Typography>
            <Typography variant="caption" sx={{ color: '#ccc', display: 'block', mt: 0.5 }}>
              {currentStep.description}
            </Typography>
            {loadingMessage && (
              <Typography variant="caption" sx={{ color: '#90caf9', display: 'block', mt: 0.5 }}>
                {loadingMessage}
              </Typography>
            )}
          </Alert>
        </Box>
      )}

      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Button
          variant="contained"
          size="large"
          onClick={handleSubmit}
          disabled={!query.trim() || isExecuting}
          startIcon={isExecuting ? <CircularProgress size={20} color="inherit" /> : <Send />}
          sx={{
            px: 4,
            py: 1.5,
            backgroundColor: isExecuting ? '#64b5f6' : '#90caf9',
            '&:hover': {
              backgroundColor: isExecuting ? '#64b5f6' : '#64b5f6',
            },
            '&:disabled': {
              backgroundColor: '#555',
              color: '#888',
            },
            ...(isExecuting && {
              animation: 'pulse 2s infinite',
              '@keyframes pulse': {
                '0%': {
                  boxShadow: '0 0 0 0 rgba(144, 202, 249, 0.7)',
                },
                '70%': {
                  boxShadow: '0 0 0 10px rgba(144, 202, 249, 0)',
                },
                '100%': {
                  boxShadow: '0 0 0 0 rgba(144, 202, 249, 0)',
                },
              },
            }),
          }}
        >
          {isExecuting ? (currentStep ? `Executing: ${currentStep.stepName}` : 'Analyzing...') : 'Analyze'}
        </Button>

        <Typography variant="body2" sx={{ color: isExecuting ? '#90caf9' : '#888' }}>
          {isExecuting ? (currentStep ? `Step: ${currentStep.stepName}` : 'Analysis in progress...') : 'Press Enter to submit'}
        </Typography>
      </Box>

      <Box>
        <Typography variant="body2" sx={{ color: '#aaa', mb: 1 }}>
          Example queries:
        </Typography>
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
          {exampleQueries.map((exampleQuery, index) => (
            <Chip
              key={index}
              label={exampleQuery}
              variant="outlined"
              size="small"
              icon={<Lightbulb />}
              onClick={() => handleExampleClick(exampleQuery)}
              disabled={isExecuting}
              sx={{
                color: '#90caf9',
                borderColor: '#90caf9',
                '&:hover': {
                  backgroundColor: '#1a3a5f',
                  borderColor: '#64b5f6',
                },
                '&:disabled': {
                  color: '#666',
                  borderColor: '#444',
                },
              }}
            />
          ))}
        </Stack>
      </Box>
    </Paper>
  );
};
