import React from 'react';
import { Box, Chip, keyframes } from '@mui/material';
import CircleIcon from '@mui/icons-material/Circle';

interface ConnectionStatusProps {
  isConnected: boolean;
}

// Pulse animation for connected indicator
const pulse = keyframes`
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
`;

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ isConnected }) => {
  return (
    <Chip
      label={isConnected ? 'Backend Online' : 'Backend Offline'}
      deleteIcon={
        <CircleIcon
          sx={{
            fontSize: 6,
            animation: isConnected ? `${pulse} 2s ease-in-out infinite` : 'none',
          }}
        />
      }
      onDelete={() => {}} // Required for deleteIcon to show
      size="small"
      sx={{
        backgroundColor: 'action.hover',
        border: '1px solid',
        borderColor: 'divider',
        color: 'text.primary',
        fontSize: '0.7rem',
        fontWeight: 500,
        height: 'auto',
        '& .MuiChip-label': {
          px: 1,
          py: 0.5,
        },
        '& .MuiChip-deleteIcon': {
          color: isConnected ? '#10b981' : '#ef4444',
          marginRight: 0.5,
          '&:hover': {
            color: isConnected ? '#10b981' : '#ef4444',
          },
        },
      }}
    />
  );
};
