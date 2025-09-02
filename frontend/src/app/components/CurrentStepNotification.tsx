import { useSelector } from 'react-redux';
import { 
  Snackbar,
  Alert,
  Box,
  Typography
} from '@mui/material';
import { 
  selectIsExecuting, 
  selectCurrentStep
} from '../../redux/selectors';

export const CurrentStepNotification = () => {
  const isExecuting = useSelector(selectIsExecuting);
  const currentStep = useSelector(selectCurrentStep);

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
        </Box>
      </Alert>
    </Snackbar>
  );
};
