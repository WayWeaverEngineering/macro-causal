import { useSelector } from 'react-redux';
import { 
  Box, 
  Paper, 
  Typography,
  Collapse,
} from '@mui/material';
import { 
  Error,
  TrendingUp
} from '@mui/icons-material';
import { 
  selectIsExecuting,
  selectCurrentStep,
  selectIsOutOfScope,
  selectOutOfScopeReason
} from '../../redux/selectors';

export const AnalysisStatus = () => {
  const isExecuting = useSelector(selectIsExecuting);
  const currentStep = useSelector(selectCurrentStep);
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
          </>
        )}
      </Paper>
    </Collapse>
  );
};
