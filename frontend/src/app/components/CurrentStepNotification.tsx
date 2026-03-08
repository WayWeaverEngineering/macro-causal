import { useSelector } from 'react-redux';
import { 
  Box, 
  Paper, 
  Typography, 
  Chip,
  Fade,
  Slide
} from '@mui/material';
import { 
  Speed,
  PlayArrow,
  CheckCircle,
  Error,
  Schedule
} from '@mui/icons-material';
import { selectIsExecuting, selectCurrentStep, selectLoadingMessage } from '../../redux/selectors';

const getStepIcon = (status: string) => {
  switch (status) {
    case 'completed':
      return <CheckCircle sx={{ color: '#4caf50', fontSize: 16 }} />;
    case 'in_progress':
      return <PlayArrow sx={{ color: '#90caf9', fontSize: 16 }} />;
    case 'failed':
      return <Error sx={{ color: '#f44336', fontSize: 16 }} />;
    default:
      return <Schedule sx={{ color: '#888', fontSize: 16 }} />;
  }
};

export const CurrentStepNotification = () => {
  const isExecuting = useSelector(selectIsExecuting);
  const currentStep = useSelector(selectCurrentStep);
  const loadingMessage = useSelector(selectLoadingMessage);

  if (!isExecuting || !currentStep) {
    return null;
  }

  return (
    <Fade in={true} timeout={500}>
      <Box
        sx={{
          position: 'fixed',
          top: 20,
          right: 20,
          zIndex: 1300,
          maxWidth: 400,
          minWidth: 300,
        }}
      >
        <Slide direction="left" in={true} timeout={300}>
          <Paper
            sx={{
              p: 2,
              backgroundColor: '#1a3a5f',
              border: '2px solid #90caf9',
              borderRadius: 2,
              boxShadow: '0 4px 20px rgba(144, 202, 249, 0.3)',
              position: 'relative',
              '&::before': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '3px',
                backgroundColor: '#90caf9',
                animation: 'pulse 2s infinite',
                '@keyframes pulse': {
                  '0%': { opacity: 1 },
                  '50%': { opacity: 0.5 },
                  '100%': { opacity: 1 },
                },
              },
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
              <Speed sx={{ color: '#90caf9', mt: 0.5, flexShrink: 0 }} />
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <Typography variant="subtitle2" sx={{ color: '#fff', fontWeight: 600 }}>
                    Current Step
                  </Typography>
                  <Chip
                    label="LIVE"
                    size="small"
                    sx={{
                      backgroundColor: '#90caf9',
                      color: '#000',
                      fontSize: '0.7rem',
                      height: '18px',
                      fontWeight: 600,
                    }}
                  />
                </Box>
                
                <Typography variant="body2" sx={{ color: '#fff', fontWeight: 500, mb: 0.5 }}>
                  {currentStep.stepName}
                </Typography>
                
                <Typography variant="caption" sx={{ color: '#ccc', display: 'block', mb: 1 }}>
                  {currentStep.description}
                </Typography>
                
                {loadingMessage && (
                  <Typography variant="caption" sx={{ color: '#90caf9', display: 'block' }}>
                    {loadingMessage}
                  </Typography>
                )}
                
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                  {getStepIcon(currentStep.status)}
                  <Typography variant="caption" sx={{ color: '#aaa', textTransform: 'capitalize' }}>
                    {currentStep.status.replace('_', ' ')}
                  </Typography>
                </Box>
              </Box>
            </Box>
          </Paper>
        </Slide>
      </Box>
    </Fade>
  );
};
