import { Box, Button, Container, Typography } from '@mui/material';
import EmailIcon from '@mui/icons-material/Email';
import DescriptionIcon from '@mui/icons-material/Description';

export const AboutMePage = () => {
  const emailAddress = 'harrynguyen92@zohomail.com';
  const resumeUrl = 'https://standardresume.co/r/4BnAeLw04dA6bEIfwAfKO';

  return (
    <Container
      maxWidth="lg"
      sx={{
        width: '100%',
        color: 'white',
        borderRadius: 2,
        py: 4,
        px: 6,
      }}
    >
      <Box display="flex" flexDirection="column" alignItems="center" textAlign="center">
        <Typography variant="h4" gutterBottom sx={{ color: '#fff' }}>
          Harry Nguyen
        </Typography>
        <Typography variant="subtitle1" gutterBottom sx={{ color: '#90caf9', fontStyle: 'italic' }}>
          Agentic AI & ML Full-Stack Engineer
        </Typography>
        <Box display="flex" justifyContent="center" alignItems="center" gap={2} sx={{ mb: 2 }}>
          <Button
            startIcon={<EmailIcon />}
            href={`mailto:${emailAddress}`}
            sx={{
              color: '#90caf9',
              borderColor: '#90caf9',
              '&:hover': {
                backgroundColor: '#1a3a5f',
                borderColor: '#64b5f6',
              },
            }}
          >
            {emailAddress}
          </Button>
          <Button
            endIcon={<DescriptionIcon />}
            href={resumeUrl}
            target="_blank"
            rel="noopener noreferrer"
            sx={{
              color: '#90caf9',
              borderColor: '#90caf9',
              '&:hover': {
                backgroundColor: '#1a3a5f',
                borderColor: '#64b5f6',
              },
            }}
          >
            My Resume
          </Button>
        </Box>
        <Typography
          variant="body1"
          gutterBottom
          sx={{ color: '#ccc', textAlign: 'justify', maxWidth: '100%', mb: 2 }}
        >
          I'm an Agentic AI & ML Engineer focused on building automated research and investment intelligence systems for institutional finance. My work centers on integrating large-scale financial data—SEC filings, macroeconomic time series, and market datasets—with LLMs, custom ML models, and structured multi-step reasoning agents to support rigorous analytical workflows.
        </Typography>
        <Typography
          variant="body1"
          gutterBottom
          sx={{ color: '#ccc', textAlign: 'justify', maxWidth: '100%', mb: 2 }}
        >
          I've designed causal inference models to quantify the effects of macroeconomic shocks, built large-scale RAG systems indexing tens of gigabytes of financial filings for cross-company and temporal analysis, and engineered agentic research pipelines capable of evidence retrieval, structured synthesis, claim validation, and scenario comparison. Across projects, I prioritize signal extraction, explainability, reproducibility, and system robustness over superficial generative outputs.
        </Typography>
        <Typography
          variant="body1"
          gutterBottom
          sx={{ color: '#ccc', textAlign: 'justify', maxWidth: '100%', mb: 2 }}
        >
          As the sole AI engineer at a fintech startup serving buy-side firms, I architect production-grade research infrastructure on AWS—owning model design, retrieval optimization, agent orchestration, and scalable deployment. My goal is to build systems that augment analysts and PMs by automating repeatable research tasks, structuring unstructured data, and enabling faster, more disciplined investment decision-making.
        </Typography>
      </Box>
    </Container>
  );
};
