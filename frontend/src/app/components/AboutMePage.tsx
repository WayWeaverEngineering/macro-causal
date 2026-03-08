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
          I build production-grade AI systems that help portfolio managers and investment analysts move faster without sacrificing rigor by compressing research cycles, improving traceability, and making AI outputs trustworthy enough for high-stakes institutional decisions. My focus is not generic GenAI experimentation, but turning fragmented, high-friction research processes into reusable, PM-facing systems that actually get adopted.
        </Typography>
        <Typography
          variant="body1"
          gutterBottom
          sx={{ color: '#ccc', textAlign: 'justify', maxWidth: '100%', mb: 2 }}
        >
          As the sole AI engineer at a fintech startup serving institutional buy-side firms — hedge funds, PE firms, and investment banks — I own most of the company's AI product architecture on AWS. I build institutional research agents, retrieval-backed fact-checking systems, and agentic financial assistants that surface relevant information from unstructured data and automate repeatable analytical work. I also own the evaluation and reliability layer that keeps these systems measurable and production-ready as they scale.
        </Typography>
        <Typography
          variant="body1"
          gutterBottom
          sx={{ color: '#ccc', textAlign: 'justify', maxWidth: '100%', mb: 2 }}
        >
          My background combines deep AI/ML engineering with end-to-end product execution across the full stack — from LLM architecture, RAG pipelines, and agentic workflows through backend APIs, cloud infrastructure, and user-facing React applications. That full-stack ownership is deliberate: systems only create value if users trust them, and trust requires getting the UI, the reasoning trace, and the underlying reliability right simultaneously. Across everything I build, I optimize for business impact, workflow adoption, and disciplined system design — not model novelty.
        </Typography>
      </Box>
    </Container>
  );
};
