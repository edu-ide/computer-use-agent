import React from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';

interface ProcessingIndicatorProps {
  isAgentProcessing: boolean;
}

export const ProcessingIndicator: React.FC<ProcessingIndicatorProps> = ({ isAgentProcessing }) => {
  if (!isAgentProcessing) return null;

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        px: 2,
        py: 1,
        borderRadius: 2,
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(0, 0, 0, 0.1)',
      }}
    >
      <CircularProgress size={20} thickness={4} />
      <Typography variant="body2" sx={{ fontWeight: 600, color: 'text.primary' }}>
        Agent is running...
      </Typography>
    </Box>
  );
};
