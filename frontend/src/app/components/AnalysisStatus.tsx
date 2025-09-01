import { useSelector } from 'react-redux';
import { 
  Box, 
  Paper, 
  Typography, 
  LinearProgress,
  Collapse,
  Stack,
  Chip
} from '@mui/material';
import { 
  CheckCircle,
  Error,
  HourglassEmpty,
  TrendingUp
} from '@mui/icons-material';
import { 
  selectIsExecuting, 
  selectExecutionSteps, 
  selectCurrentStep,
  selectProgressPercentage,
  selectIsOutOfScope,
  selectOutOfScopeReason
} from '../../redux/selectors';

export const AnalysisStatus = () => {
  const isExecuting = useSelector(selectIsExecuting);
  const executionSteps = useSelector(selectExecutionSteps);
  const currentStep = useSelector(selectCurrentStep);
  const progressPercentage = useSelector(selectProgressPercentage);
  const isOutOfScope = useSelector(selectIsOutOfScope);
  const outOfScopeReason = useSelector(selectOutOfScopeReason);

  if (!isExecuting && !isOutOfScope) {
    return null;
  }

  return (
    <Collapse in={isExecuting || isOutOfScope}>
      <Paper sx={{ p: 3, mb: 3 }}>
        {isOutOfScope ? (
          <Box sx={{ textAlign: 'center' }}>
            <Error sx={{ fontSize: 48, color: '#ff9800', mb: 2 }} />
            <Typography variant="h6" sx={{ color: '#fff', mb: 1 }}>
              Query Out of Scope
            </Typography>
            <Typography variant="body1" sx={{ color: '#ccc', mb: 2 }}>
              {outOfScopeReason || 'This query is not suitable for macroeconomic causal analysis.'}
            </Typography>
            <Typography variant="body2" sx={{ color: '#aaa' }}>
              Please try asking about macroeconomic variables, monetary policy, inflation, GDP, 
              interest rates, or their effects on asset returns.
            </Typography>
          </Box>
        ) : (
          <>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <TrendingUp sx={{ color: '#90caf9', mr: 1 }} />
              <Typography variant="h6" sx={{ color: '#fff' }}>
                Causal Analysis Progress
              </Typography>
            </Box>

            {/* Progress Bar */}
            <Box sx={{ mb: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" sx={{ color: '#ccc' }}>
                  Progress
                </Typography>
                <Typography variant="body2" sx={{ color: '#90caf9', fontWeight: 500 }}>
                  {progressPercentage}%
                </Typography>
              </Box>
              <LinearProgress 
                variant="determinate" 
                value={progressPercentage} 
                sx={{
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: '#333',
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: '#90caf9',
                    borderRadius: 4,
                  },
                }}
              />
            </Box>

            {/* Current Step */}
            {currentStep && (
              <Box sx={{ mb: 3 }}>
                <Typography variant="body1" sx={{ color: '#fff', mb: 1, fontWeight: 500 }}>
                  Current Step: {currentStep.stepName}
                </Typography>
                <Typography variant="body2" sx={{ color: '#ccc' }}>
                  {currentStep.description}
                </Typography>
              </Box>
            )}

            {/* Execution Steps */}
            {executionSteps.length > 0 && (
              <Box>
                <Typography variant="body1" sx={{ color: '#fff', mb: 2, fontWeight: 500 }}>
                  Execution Steps:
                </Typography>
                <Stack spacing={1}>
                  {executionSteps.map((step) => (
                    <Box 
                      key={step.stepId} 
                      sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        p: 1.5, 
                        backgroundColor: '#2a2a2a',
                        borderRadius: 1,
                        border: '1px solid #444'
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
                        {step.status === 'completed' && (
                          <CheckCircle sx={{ color: '#4caf50', fontSize: 20, mr: 1 }} />
                        )}
                        {step.status === 'in_progress' && (
                          <HourglassEmpty sx={{ color: '#90caf9', fontSize: 20, mr: 1 }} />
                        )}
                        {step.status === 'failed' && (
                          <Error sx={{ color: '#f44336', fontSize: 20, mr: 1 }} />
                        )}
                        {step.status === 'pending' && (
                          <Box sx={{ width: 20, height: 20, borderRadius: '50%', backgroundColor: '#666', mr: 1 }} />
                        )}
                      </Box>
                      
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="body2" sx={{ color: '#fff', fontWeight: 500 }}>
                          {step.stepName}
                        </Typography>
                        <Typography variant="caption" sx={{ color: '#aaa' }}>
                          {step.description}
                        </Typography>
                      </Box>

                      <Chip 
                        label={step.status.replace('_', ' ')} 
                        size="small"
                        sx={{
                          backgroundColor: 
                            step.status === 'completed' ? '#4caf50' :
                            step.status === 'in_progress' ? '#90caf9' :
                            step.status === 'failed' ? '#f44336' : '#666',
                          color: '#fff',
                          textTransform: 'capitalize',
                          fontSize: '0.7rem',
                        }}
                      />
                    </Box>
                  ))}
                </Stack>
              </Box>
            )}
          </>
        )}
      </Paper>
    </Collapse>
  );
};
