import { useSelector } from 'react-redux';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  Box,
  Typography,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Stack,
} from '@mui/material';
import {
  ExpandMore,
  CheckCircle,
  Error,
  Schedule,
  PlayArrow,
  Psychology,
  Science,
  Article,
} from '@mui/icons-material';
import { selectExecutionSteps } from '../../redux/selectors';
import type { ExecutionStep } from '../../models/analysis';

const getStepIcon = (status: string) => {
  switch (status) {
    case 'completed':
      return <CheckCircle sx={{ color: '#4caf50', fontSize: 20 }} />;
    case 'in_progress':
      return <PlayArrow sx={{ color: '#90caf9', fontSize: 20 }} />;
    case 'failed':
      return <Error sx={{ color: '#f44336', fontSize: 20 }} />;
    default:
      return <Schedule sx={{ color: '#888', fontSize: 20 }} />;
  }
};

const renderStepMetadata = (step: ExecutionStep) => {
  const meta = step.metadata;
  if (!meta || Object.keys(meta).length === 0) return null;

  const items: { label: string; content: React.ReactNode; icon?: React.ReactNode }[] = [];

  // Query (step_1)
  if (meta.query) {
    items.push({
      label: 'User Query',
      content: <Typography variant="body2" sx={{ color: '#ccc', lineHeight: 1.6 }}>{meta.query}</Typography>,
      icon: <Psychology sx={{ fontSize: 18, color: '#90caf9', mr: 0.5 }} />,
    });
  }

  // Model inputs (step_1, step_2)
  if (meta.modelInputs) {
    items.push({
      label: 'Model Inputs',
      content: (
        <Typography variant="body2" sx={{ color: '#888', fontFamily: 'monospace', fontSize: '0.8rem', whiteSpace: 'pre-wrap' }}>
          {typeof meta.modelInputs === 'string' ? meta.modelInputs : JSON.stringify(meta.modelInputs, null, 2)}
        </Typography>
      ),
      icon: <Science sx={{ fontSize: 18, color: '#90caf9', mr: 0.5 }} />,
    });
  }

  // Model results (step_2, step_3)
  if (meta.modelResults) {
    items.push({
      label: 'Model Results',
      content: (
        <Typography variant="body2" sx={{ color: '#888', fontFamily: 'monospace', fontSize: '0.8rem', whiteSpace: 'pre-wrap' }}>
          {typeof meta.modelResults === 'string' ? meta.modelResults : JSON.stringify(meta.modelResults, null, 2)}
        </Typography>
      ),
      icon: <Science sx={{ fontSize: 18, color: '#90caf9', mr: 0.5 }} />,
    });
  }

  // Final response (step_3)
  if (meta.finalResponse) {
    items.push({
      label: 'Generated Response',
      content: (
        <Typography variant="body2" sx={{ color: '#ccc', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
          {meta.finalResponse}
        </Typography>
      ),
      icon: <Article sx={{ fontSize: 18, color: '#90caf9', mr: 0.5 }} />,
    });
  }

  // Generic fallback for other metadata
  const knownKeys = new Set(['query', 'modelInputs', 'modelResults', 'finalResponse']);
  const otherKeys = Object.keys(meta).filter(k => !knownKeys.has(k));
  if (otherKeys.length > 0) {
    items.push({
      label: 'Other',
      content: (
        <Typography variant="body2" sx={{ color: '#888', fontFamily: 'monospace', fontSize: '0.8rem' }}>
          {JSON.stringify(Object.fromEntries(otherKeys.map(k => [k, meta[k]])), null, 2)}
        </Typography>
      ),
    });
  }

  return items.length === 0 ? null : (
    <Stack spacing={2} sx={{ mt: 1 }}>
      {items.map((item, idx) => (
        <Box key={idx}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
            {item.icon}
            <Typography variant="caption" sx={{ color: '#aaa', fontWeight: 600, textTransform: 'uppercase' }}>
              {item.label}
            </Typography>
          </Box>
          {item.content}
        </Box>
      ))}
    </Stack>
  );
};

export interface ExecutionDetailsModalProps {
  open: boolean;
  onClose: () => void;
}

export const ExecutionDetailsModal = ({ open, onClose }: ExecutionDetailsModalProps) => {
  const executionSteps = useSelector(selectExecutionSteps);

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          backgroundColor: '#1e1e1e',
          border: '1px solid #444',
        },
      }}
    >
      <DialogTitle sx={{ color: '#fff', borderBottom: '1px solid #444' }}>
        Execution Details
      </DialogTitle>
      <DialogContent sx={{ pt: 2 }}>
        {executionSteps.length === 0 ? (
          <Typography variant="body2" sx={{ color: '#888' }}>
            No execution steps available.
          </Typography>
        ) : (
          <Stack spacing={1}>
            {executionSteps.map((step) => (
              <Accordion
                key={step.stepId}
                sx={{
                  backgroundColor: '#2a2a2a',
                  border: '1px solid #444',
                  '&:before': { display: 'none' },
                }}
              >
                <AccordionSummary expandIcon={<ExpandMore sx={{ color: '#fff' }} />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                    {getStepIcon(step.status)}
                    <Typography variant="body2" sx={{ color: '#fff', fontWeight: 500 }}>
                      {step.stepName}
                    </Typography>
                    <Chip
                      label={step.status.replace('_', ' ')}
                      size="small"
                      sx={{
                        backgroundColor: step.status === 'completed' ? '#4caf50' : step.status === 'failed' ? '#f44336' : '#666',
                        color: '#fff',
                        textTransform: 'capitalize',
                      }}
                    />
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Typography variant="caption" sx={{ color: '#aaa', display: 'block', mb: 1 }}>
                    {step.description}
                  </Typography>
                  {step.startTime && (
                    <Typography variant="caption" sx={{ color: '#666', display: 'block', mb: 1 }}>
                      Started: {new Date(step.startTime).toLocaleString()}
                    </Typography>
                  )}
                  {step.error && (
                    <Typography variant="body2" sx={{ color: '#f44336', mb: 1 }}>
                      Error: {step.error}
                    </Typography>
                  )}
                  {renderStepMetadata(step)}
                </AccordionDetails>
              </Accordion>
            ))}
          </Stack>
        )}
      </DialogContent>
    </Dialog>
  );
};
