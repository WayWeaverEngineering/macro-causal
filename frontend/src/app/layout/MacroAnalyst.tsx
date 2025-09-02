import { CssBaseline, ThemeProvider, createTheme, Container, Box, Button, Dialog, DialogTitle, DialogContent } from '@mui/material';
import { QueryInput } from '../components/QueryInput';
import { AnalysisStatus } from '../components/AnalysisStatus';
import { CausalAnalysisResults } from '../components/CausalAnalysisResults';
import { Header } from '../components/Header';
import { CurrentStepNotification } from '../components/CurrentStepNotification';
import { useState } from 'react';
import PipelineSvg from '../images/ml-pipeline-success.svg';

const darkTheme = createTheme({
  palette: { 
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: '#1e1e1e',
          border: '1px solid #333',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            '& fieldset': {
              borderColor: '#444',
            },
            '&:hover fieldset': {
              borderColor: '#666',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#90caf9',
            },
          },
        },
      },
    },
  },
});

function MacroAnalyst() {
  const [openPipelineDialog, setOpenPipelineDialog] = useState(false);

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', backgroundColor: '#121212' }}>
        <Header />
        <Container maxWidth="lg" sx={{ py: 4 }}>
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
            <Button variant="outlined" color="primary" onClick={() => setOpenPipelineDialog(true)}>
              View ML Pipeline Design
            </Button>
          </Box>
          <QueryInput />
          <AnalysisStatus />
          <CausalAnalysisResults />
        </Container>
        <CurrentStepNotification />

        <Dialog
          open={openPipelineDialog}
          onClose={() => setOpenPipelineDialog(false)}
          maxWidth="lg"
          fullWidth
          PaperProps={{
            sx: {
              backgroundColor: '#0e0e0e',
            }
          }}
        >
          <DialogTitle sx={{ backgroundColor: '#0e0e0e' }}>ML Pipeline Design</DialogTitle>
          <DialogContent dividers sx={{ backgroundColor: '#0b0b0b' }}>
            <Box sx={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
              <img
                src={PipelineSvg}
                alt="ML Pipeline Design"
                style={{ maxWidth: '100%', maxHeight: '70vh', height: 'auto' }}
              />
            </Box>
          </DialogContent>
        </Dialog>
      </Box>
    </ThemeProvider>
  );
}

export default MacroAnalyst;
