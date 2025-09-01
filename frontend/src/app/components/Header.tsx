import { Box, Typography, Paper } from '@mui/material';
import { Analytics } from '@mui/icons-material';

export const Header = () => {
  return (
    <Paper 
      elevation={0} 
      sx={{ 
        backgroundColor: '#1a1a1a', 
        borderBottom: '1px solid #333',
        borderRadius: 0,
        py: 3
      }}
    >
      <Box sx={{ textAlign: 'center' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
          <Analytics sx={{ fontSize: 40, color: '#90caf9', mr: 2, display: { xs: 'none', sm: 'inline-flex' } }} />
          <Typography variant="h4" component="h1" sx={{ color: '#fff', fontWeight: 600 }}>
            Macro Causal Inference AI Analyst
          </Typography>
        </Box>
        <Typography variant="body1" sx={{ color: '#aaa', maxWidth: 600, mx: 'auto' }}>
          Ask questions about macroeconomic relationships and get AI-powered causal analysis using X-Learner 
          and Regime Classifier models. Understand true causal effects, not just correlations.
        </Typography>
      </Box>
    </Paper>
  );
};
