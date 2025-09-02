import { useSelector } from 'react-redux';
import { 
  Box, 
  Paper, 
  Typography, 
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  Collapse
} from '@mui/material';
import { 
  ExpandMore,
  Summarize,
  Lightbulb,
  Warning,
  Psychology,
  CheckCircle,
  Timeline,
  Assessment,
  Science
} from '@mui/icons-material';
import { 
  selectCurrentAnalysis, 
  selectHasResults
} from '../../redux/selectors';

const formatConfidence = (confidence: number) => {
  return `${Math.round(confidence * 100)}%`;
};

const getConfidenceColor = (confidence: number) => {
  if (confidence >= 0.8) return '#4caf50';
  if (confidence >= 0.6) return '#ff9800';
  return '#f44336';
};



const formatEffect = (effect: number) => {
  const sign = effect >= 0 ? '+' : '';
  return `${sign}${(effect * 100).toFixed(2)}%`;
};

const getEffectColor = (effect: number) => {
  if (effect > 0) return '#4caf50';
  if (effect < 0) return '#f44336';
  return '#ff9800';
};

export const CausalAnalysisResults = () => {
  const analysis = useSelector(selectCurrentAnalysis);
  const hasResults = useSelector(selectHasResults);

  if (!hasResults) {
    return null;
  }

  return (
    <Collapse in={hasResults}>
      <Paper sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <CheckCircle sx={{ color: '#4caf50', mr: 1 }} />
          <Typography variant="h5" sx={{ color: '#fff', fontWeight: 600 }}>
            Causal Analysis Results
          </Typography>
        </Box>

        {/* Causal Effect Summary */}
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Science sx={{ color: '#90caf9', mr: 1 }} />
            <Typography variant="h6" sx={{ color: '#fff' }}>
              Causal Effect
            </Typography>
          </Box>
          <Paper sx={{ p: 2, backgroundColor: '#2a2a2a', border: '1px solid #444' }}>
                          <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 2 }}>
                <Box sx={{ flex: 1, textAlign: 'center' }}>
                  <Typography variant="h4" sx={{ color: getEffectColor(analysis?.causalEffect?.effect ?? 0), fontWeight: 600 }}>
                    {formatEffect(analysis?.causalEffect?.effect ?? 0)}
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#aaa' }}>
                    Causal Effect
                  </Typography>
                </Box>
                <Box sx={{ flex: 1, textAlign: 'center' }}>
                  <Typography variant="h4" sx={{ color: getConfidenceColor(analysis?.uncertainty?.confidence ?? 0), fontWeight: 600 }}>
                    {formatConfidence(analysis?.uncertainty?.confidence ?? 0)}
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#aaa' }}>
                    Confidence Level
                  </Typography>
                </Box>
              </Box>
            <Box sx={{ mt: 2, textAlign: 'center' }}>
              <Typography variant="body2" sx={{ color: '#ccc' }}>
                Confidence Interval: [{formatEffect(analysis?.causalEffect?.confidenceInterval?.[0] ?? 0)}, {formatEffect(analysis?.causalEffect?.confidenceInterval?.[1] ?? 0)}]
              </Typography>
              <Typography variant="body2" sx={{ color: '#ccc' }}>
                P-value: {(analysis?.causalEffect?.pValue ?? 0).toFixed(4)}
              </Typography>
            </Box>
          </Paper>
        </Box>

        {/* Summary */}
        <Accordion defaultExpanded sx={{ mb: 2, backgroundColor: '#2a2a2a', border: '1px solid #444' }}>
          <AccordionSummary expandIcon={<ExpandMore sx={{ color: '#fff' }} />}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Summarize sx={{ color: '#90caf9', mr: 1 }} />
              <Typography variant="h6" sx={{ color: '#fff' }}>
                Summary
              </Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="body1" sx={{ color: '#ccc', lineHeight: 1.6 }}>
              {analysis?.summary || 'No summary available'}
            </Typography>
          </AccordionDetails>
        </Accordion>

        {/* Key Insights */}
        {analysis?.keyInsights && analysis.keyInsights.length > 0 && (
          <Accordion defaultExpanded sx={{ mb: 2, backgroundColor: '#2a2a2a', border: '1px solid #444' }}>
            <AccordionSummary expandIcon={<ExpandMore sx={{ color: '#fff' }} />}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Lightbulb sx={{ color: '#f48fb1', mr: 1 }} />
                <Typography variant="h6" sx={{ color: '#fff' }}>
                  Key Insights
                </Typography>
                <Chip 
                  label={analysis.keyInsights.length} 
                  size="small" 
                  sx={{ ml: 1, backgroundColor: '#f48fb1', color: '#fff' }}
                />
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <List sx={{ p: 0 }}>
                {analysis.keyInsights.map((insight, index) => (
                  <ListItem key={index} sx={{ px: 0, py: 1 }}>
                    <ListItemText
                      primary={
                        <Typography variant="body1" sx={{ color: '#ccc', lineHeight: 1.5 }}>
                          • {insight || 'No insight available'}
                        </Typography>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </AccordionDetails>
          </Accordion>
        )}

        {/* Regime Analysis */}
        {analysis?.regimeAnalysis && (
          <Accordion defaultExpanded sx={{ mb: 2, backgroundColor: '#2a2a2a', border: '1px solid #444' }}>
            <AccordionSummary expandIcon={<ExpandMore sx={{ color: '#fff' }} />}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Timeline sx={{ color: '#4caf50', mr: 1 }} />
                <Typography variant="h6" sx={{ color: '#fff' }}>
                  Market Regime Analysis
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body1" sx={{ color: '#fff', mb: 1, fontWeight: 500 }}>
                  Current Regime: {analysis.regimeAnalysis.regimeNames?.[analysis.regimeAnalysis.currentRegime] || `Regime ${analysis.regimeAnalysis.currentRegime}`}
                </Typography>
                <Typography variant="body2" sx={{ color: '#ccc', mb: 2 }}>
                  The system identifies different market states where causal relationships behave differently.
                </Typography>
                <Box sx={{ backgroundColor: '#1a1a1a', border: '1px solid #333', p: 2, borderRadius: 1 }}>
                  <Typography variant="body2" sx={{ color: '#aaa', mb: 1, fontWeight: 500 }}>
                    Regime definitions (by volatility context):
                  </Typography>
                  <List sx={{ p: 0 }}>
                    <ListItem sx={{ px: 0, py: 0.5 }}>
                      <ListItemText
                        primary={
                          <Typography variant="body2" sx={{ color: '#ccc' }}>
                            • Regime 0: Low volatility conditions (calmer markets)
                          </Typography>
                        }
                      />
                    </ListItem>
                    <ListItem sx={{ px: 0, py: 0.5 }}>
                      <ListItemText
                        primary={
                          <Typography variant="body2" sx={{ color: '#ccc' }}>
                            • Regime 1: Medium volatility conditions
                          </Typography>
                        }
                      />
                    </ListItem>
                    <ListItem sx={{ px: 0, py: 0.5 }}>
                      <ListItemText
                        primary={
                          <Typography variant="body2" sx={{ color: '#ccc' }}>
                            • Regime 2: High volatility conditions (stressed markets)
                          </Typography>
                        }
                      />
                    </ListItem>
                  </List>
                </Box>
              </Box>
              
                             <Box sx={{ display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, gap: 2, flexWrap: 'wrap' }}>
                 {analysis.regimeAnalysis.regimeProbabilities?.map((prob, index) => (
                   <Box key={index} sx={{ flex: '1 1 200px', minWidth: 0 }}>
                     <Paper sx={{ p: 2, backgroundColor: '#1a1a1a', border: '1px solid #333', textAlign: 'center' }}>
                       <Typography variant="h6" sx={{ color: '#90caf9' }}>
                         {Math.round(prob * 100)}%
                       </Typography>
                       <Typography variant="caption" sx={{ color: '#aaa' }}>
                         {analysis.regimeAnalysis.regimeNames?.[index] || `Regime ${index}`}
                       </Typography>
                     </Paper>
                   </Box>
                 ))}
               </Box>
            </AccordionDetails>
          </Accordion>
        )}

        {/* Uncertainty Analysis */}
        {analysis?.uncertainty && (
          <Accordion sx={{ mb: 2, backgroundColor: '#2a2a2a', border: '1px solid #444' }}>
            <AccordionSummary expandIcon={<ExpandMore sx={{ color: '#fff' }} />}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Assessment sx={{ color: '#ff9800', mr: 1 }} />
                <Typography variant="h6" sx={{ color: '#fff' }}>
                  Uncertainty Analysis
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body1" sx={{ color: '#fff', mb: 1 }}>
                  Uncertainty Level: {analysis.uncertainty.uncertainty.toFixed(3)}
                </Typography>
                <Typography variant="body2" sx={{ color: '#ccc', mb: 2 }}>
                  Reliability: {analysis.uncertainty.reliability}
                </Typography>
              </Box>
              
              {analysis.uncertainty.factors && analysis.uncertainty.factors.length > 0 && (
                <Box>
                  <Typography variant="body2" sx={{ color: '#aaa', mb: 1, fontWeight: 500 }}>
                    Uncertainty Factors:
                  </Typography>
                  <List sx={{ p: 0 }}>
                    {analysis.uncertainty.factors.map((factor, index) => (
                      <ListItem key={index} sx={{ px: 0, py: 0.5 }}>
                        <ListItemText
                          primary={
                            <Typography variant="body2" sx={{ color: '#ccc' }}>
                              • {factor}
                            </Typography>
                          }
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
            </AccordionDetails>
          </Accordion>
        )}

        {/* Methodology */}
        {analysis?.methodology && (
          <Accordion sx={{ mb: 2, backgroundColor: '#2a2a2a', border: '1px solid #444' }}>
            <AccordionSummary expandIcon={<ExpandMore sx={{ color: '#fff' }} />}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Psychology sx={{ color: '#9c27b0', mr: 1 }} />
                <Typography variant="h6" sx={{ color: '#fff' }}>
                  Methodology
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body1" sx={{ color: '#ccc', lineHeight: 1.6 }}>
                {analysis.methodology}
              </Typography>
            </AccordionDetails>
          </Accordion>
        )}

        {/* Limitations */}
        {analysis?.limitations && analysis.limitations.length > 0 && (
          <Accordion sx={{ backgroundColor: '#2a2a2a', border: '1px solid #444' }}>
            <AccordionSummary expandIcon={<ExpandMore sx={{ color: '#fff' }} />}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Warning sx={{ color: '#ff9800', mr: 1 }} />
                <Typography variant="h6" sx={{ color: '#fff' }}>
                  Limitations
                </Typography>
                <Chip 
                  label={analysis.limitations.length} 
                  size="small" 
                  sx={{ ml: 1, backgroundColor: '#ff9800', color: '#fff' }}
                />
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <List sx={{ p: 0 }}>
                {analysis.limitations.map((limitation, index) => (
                  <ListItem key={index} sx={{ px: 0, py: 1 }}>
                    <ListItemText
                      primary={
                        <Typography variant="body2" sx={{ color: '#ccc', lineHeight: 1.5 }}>
                          • {limitation || 'No limitation details available'}
                        </Typography>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </AccordionDetails>
          </Accordion>
        )}
      </Paper>
    </Collapse>
  );
};
