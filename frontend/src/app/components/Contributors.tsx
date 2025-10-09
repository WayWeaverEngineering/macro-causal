import { 
  Box,
  Typography, 
  Card, 
  CardContent, 
  Chip, 
  Container,
  Button,
  Stack
} from '@mui/material';
import { 
  LinkedIn,
  Description
} from '@mui/icons-material';
import { contributors, Contributor } from '../../models/contributors';

export const Contributors = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>

      {/* Contributors Grid */}
      <Box 
        sx={{ 
          display: 'grid',
          gridTemplateColumns: {
            xs: '1fr',
            sm: 'repeat(2, 1fr)',
            md: 'repeat(2, 1fr)',
          },
          gap: 3,
        }}
      >
        {contributors.map((contributor: Contributor) => (
          <Card 
            key={contributor.id}
            sx={{ 
              height: '100%',
              backgroundColor: '#1e1e1e',
              border: '1px solid #333',
              transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: '0 8px 25px rgba(144, 202, 249, 0.15)',
                borderColor: '#90caf9',
              },
            }}
          >
            <CardContent sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
              {/* Name and Title Section */}
              <Box sx={{ mb: 2 }}>
                <Typography variant="h6" sx={{ color: '#fff', fontWeight: 600, mb: 0.5 }}>
                  {contributor.name}
                </Typography>
                <Typography variant="body2" sx={{ color: '#90caf9', fontWeight: 500 }}>
                  {contributor.title}
                </Typography>
              </Box>

              {/* Bio */}
              <Typography 
                variant="body2" 
                sx={{ 
                  color: '#ccc', 
                  lineHeight: 1.6, 
                  mb: 2, 
                  flex: 1,
                  display: '-webkit-box',
                  WebkitLineClamp: 4,
                  WebkitBoxOrient: 'vertical',
                  overflow: 'hidden',
                }}
              >
                {contributor.bio}
              </Typography>

              {/* Roles */}
              <Box sx={{ mb: 2 }}>
                <Typography variant="caption" sx={{ color: '#aaa', mb: 1, display: 'block' }}>
                  Key Contributions:
                </Typography>
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                  {contributor.roles.map((role) => (
                    <Chip
                      key={role}
                      label={role}
                      size="small"
                      sx={{
                        backgroundColor: '#333',
                        color: '#fff',
                        fontSize: '0.75rem',
                        height: '24px',
                        border: '1px solid #555',
                        '& .MuiChip-label': {
                          px: 1,
                        },
                      }}
                    />
                  ))}
                </Stack>
              </Box>

              {/* Action Buttons */}
              {(contributor.linkedInUrl || contributor.resumeUrl) && (
                <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 'auto' }}>
                  {contributor.linkedInUrl && (
                    <Button
                      component="a"
                      href={contributor.linkedInUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      variant="contained"
                      startIcon={<LinkedIn />}
                      sx={{
                        backgroundColor: '#0077b5',
                        color: '#fff',
                        px: 3,
                        py: 1,
                        fontSize: '0.875rem',
                        fontWeight: 600,
                        textTransform: 'none',
                        borderRadius: 2,
                        '&:hover': {
                          backgroundColor: '#005885',
                          transform: 'translateY(-1px)',
                          boxShadow: '0 4px 12px rgba(0, 119, 181, 0.3)',
                        },
                        transition: 'all 0.2s ease-in-out',
                      }}
                    >
                      LinkedIn
                    </Button>
                  )}
                  
                  {contributor.resumeUrl && (
                    <Button
                      component="a"
                      href={contributor.resumeUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      variant="contained"
                      startIcon={<Description />}
                      sx={{
                        backgroundColor: '#2e7d32',
                        color: '#fff',
                        px: 3,
                        py: 1,
                        fontSize: '0.875rem',
                        fontWeight: 600,
                        textTransform: 'none',
                        borderRadius: 2,
                        '&:hover': {
                          backgroundColor: '#1b5e20',
                          transform: 'translateY(-1px)',
                          boxShadow: '0 4px 12px rgba(46, 125, 50, 0.3)',
                        },
                        transition: 'all 0.2s ease-in-out',
                      }}
                    >
                      Resume
                    </Button>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        ))}
      </Box>
    </Container>
  );
};
