import { useSelector } from 'react-redux';
import { 
  Snackbar,
  Alert,
  Box,
  Typography,
  LinearProgress
} from '@mui/material';
import { 
  selectIsExecuting, 
  selectCurrentStep,
  selectProgressPercentage 
} from '../../redux/selectors';

export const CurrentStepNotification = () => {
  const isExecuting = useSelector(selectIsExecuting);
  const currentStep = useSelector(selectCurrentStep);
  const progressPercentage = useSelector(selectProgressPercentage);

  if (!isExecuting || !currentStep) {
    return null;
  }

  return (
    <Snackbar
      open={isExecuting && !!currentStep}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      sx={{ 
        '& .MuiAlert-root': {
          backgroundColor: '#1a3a5f',
          border: '1px solid #90caf9',
          color: '#fff',
          minWidth: 400,
        }
      }}
    >
      <Alert 
        severity="info" 
        icon={false}
        sx={{ 
          backgroundColor: '#1a3a5f',
          border: '1px solid #90caf9',
          color: '#fff',
          '& .MuiAlert-message': {
            color: '#fff',
            width: '100%',
          }
        }}
      >
        <Box sx={{ width: '100%' }}>
          <Typography variant="body1" sx={{ color: '#fff', fontWeight: 500, mb: 1 }}>
            {currentStep.stepName}
          </Typography>
          <Typography variant="body2" sx={{ color: '#ccc', mb: 2 }}>
            {currentStep.description}
          </Typography>
          
          <Box sx={{ mb: 1 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
              <Typography variant="caption" sx={{ color: '#90caf9' }}>
                Progress
              </Typography>
              <Typography variant="caption" sx={{ color: '#90caf9', fontWeight: 500 }}>
                {progressPercentage}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={progressPercentage} 
              sx={{
                height: 4,
                borderRadius: 2,
                backgroundColor: '#2a2a2a',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: '#90caf9',
                  borderRadius: 2,
                },
              }}
            />
          </Box>
        </Box>
      </Alert>
    </Snackbar>
  );
};
