import { Box, Typography, Paper, Button } from '@mui/material';
import { Analytics, People, ArrowBack } from '@mui/icons-material';

interface HeaderProps {
  onNavigateToContributors?: () => void;
  onBackToMain?: () => void;
  currentView?: 'main' | 'contributors';
}

export const Header = ({ onNavigateToContributors, onBackToMain, currentView }: HeaderProps) => {
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
        <Typography variant="body1" sx={{ color: '#aaa', maxWidth: 600, mx: 'auto', mb: 2 }}>
          Ask questions about macroeconomic relationships and get AI-powered causal analysis using X-Learner 
          and Regime Classifier models. Understand true causal effects, not just correlations.
        </Typography>
        
        {/* Navigation Button */}
        {currentView === 'main' && onNavigateToContributors && (
          <Button
            variant="outlined"
            startIcon={<People />}
            onClick={onNavigateToContributors}
            sx={{
              color: '#90caf9',
              borderColor: '#90caf9',
              '&:hover': {
                backgroundColor: '#1a3a5f',
                borderColor: '#64b5f6',
              },
            }}
          >
            Meet the Team
          </Button>
        )}
        
        {currentView === 'contributors' && onBackToMain && (
          <Button
            variant="outlined"
            startIcon={<ArrowBack />}
            onClick={onBackToMain}
            sx={{
              color: '#90caf9',
              borderColor: '#90caf9',
              '&:hover': {
                backgroundColor: '#1a3a5f',
                borderColor: '#64b5f6',
              },
            }}
          >
            Back to Analyst
          </Button>
        )}
      </Box>
    </Paper>
  );
};
