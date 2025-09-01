import { CssBaseline, ThemeProvider, createTheme, Container, Box } from '@mui/material';
import { QueryInput } from '../components/QueryInput';
import { AnalysisStatus } from '../components/AnalysisStatus';
import { CausalAnalysisResults } from '../components/CausalAnalysisResults';
import { Header } from '../components/Header';
import { CurrentStepNotification } from '../components/CurrentStepNotification';

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
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', backgroundColor: '#121212' }}>
        <Header />
        <Container maxWidth="lg" sx={{ py: 4 }}>
          <QueryInput />
          <AnalysisStatus />
          <CausalAnalysisResults />
        </Container>
        <CurrentStepNotification />
      </Box>
    </ThemeProvider>
  );
}

export default MacroAnalyst;
