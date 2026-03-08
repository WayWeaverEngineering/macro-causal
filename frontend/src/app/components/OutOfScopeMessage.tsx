import { useSelector } from 'react-redux';
import { Box, Paper, Typography, Collapse } from '@mui/material';
import { Error } from '@mui/icons-material';
import { selectIsExecuting, selectIsOutOfScope, selectOutOfScopeReason } from '../../redux/selectors';

export const OutOfScopeMessage = () => {
  const isExecuting = useSelector(selectIsExecuting);
  const isOutOfScope = useSelector(selectIsOutOfScope);
  const outOfScopeReason = useSelector(selectOutOfScopeReason);

  if (!isOutOfScope || isExecuting) {
    return null;
  }

  return (
    <Collapse in={isOutOfScope && !isExecuting}>
      <Paper sx={{ p: 3, mb: 3, backgroundColor: '#2a2a2a', border: '1px solid #444' }}>
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
      </Paper>
    </Collapse>
  );
};
