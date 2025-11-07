import React from 'react';
import { Card, CardContent, Box, Typography, CircularProgress } from '@mui/material';
import { keyframes } from '@mui/system';

// Border pulse animation
const borderPulse = keyframes`
  0%, 100% {
    border-color: rgba(79, 134, 198, 0.4);
    box-shadow: 0 2px 8px rgba(79, 134, 198, 0.15);
  }
  50% {
    border-color: rgba(79, 134, 198, 0.8);
    box-shadow: 0 2px 12px rgba(79, 134, 198, 0.3);
  }
`;

// Background pulse animation
const backgroundPulse = keyframes`
  0%, 100% {
    background-color: rgba(79, 134, 198, 0.03);
  }
  50% {
    background-color: rgba(79, 134, 198, 0.08);
  }
`;

interface ThinkingStepCardProps {
  isActive?: boolean;
}

export const ThinkingStepCard: React.FC<ThinkingStepCardProps> = ({ isActive = false }) => {

  return (
    <Card
      elevation={0}
      sx={{
        backgroundColor: 'background.paper',
        border: '1px solid',
        borderColor: (theme) => `${isActive ? theme.palette.primary.main : theme.palette.divider} !important`,
        borderRadius: 1.5,
        animation: isActive ? `${borderPulse} 2s ease-in-out infinite` : 'none',
        position: 'relative',
        overflow: 'hidden',
        boxShadow: isActive ? (theme) => `0 2px 8px ${theme.palette.mode === 'dark' ? 'rgba(79, 134, 198, 0.3)' : 'rgba(79, 134, 198, 0.2)'}` : 'none',
        transition: 'all 0.2s ease',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          animation: isActive ? `${backgroundPulse} 2s ease-in-out infinite` : 'none',
          zIndex: 0,
        },
      }}
    >
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 }, position: 'relative', zIndex: 1 }}>
        {/* Header with spinner */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {/* Spinner circulaire */}
            <CircularProgress
              size={32}
              thickness={3.5}
              sx={{
                color: 'primary.main',
              }}
            />
          </Box>

          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography
              sx={{
                fontSize: '0.85rem',
                fontWeight: 700,
                color: 'primary.main',
                lineHeight: 1.3,
              }}
            >
              Agent
            </Typography>
            <Typography
              sx={{
                fontSize: '0.7rem',
                color: 'text.secondary',
                lineHeight: 1.2,
                fontStyle: 'italic',
              }}
            >
              Thinking...
            </Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};
