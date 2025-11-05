import { FinalStep } from '@/types/agent';
import React from 'react';
import { Card, CardContent, Box, Typography } from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import { useAgentStore } from '@/stores/agentStore';

interface FinalStepCardProps {
  finalStep: FinalStep;
  isActive?: boolean;
}

export const FinalStepCard: React.FC<FinalStepCardProps> = ({ finalStep, isActive = false }) => {
  const setSelectedStepIndex = useAgentStore((state) => state.setSelectedStepIndex);

  const isSuccess = finalStep.type === 'success';

  const handleClick = () => {
    // Clicking on final step goes to live mode (null)
    setSelectedStepIndex(null);
  };

  return (
    <Card
      elevation={0}
      onClick={handleClick}
      sx={{
        backgroundColor: 'background.paper',
        border: '1px solid',
        borderColor: (theme) => `${isActive
          ? isSuccess ? theme.palette.success.main : theme.palette.error.main
          : theme.palette.divider} !important`,
        borderRadius: 1.5,
        transition: 'all 0.2s ease',
        cursor: 'pointer',
        boxShadow: isActive
          ? (theme) => isSuccess
            ? `0 2px 8px ${theme.palette.mode === 'dark' ? 'rgba(102, 187, 106, 0.3)' : 'rgba(102, 187, 106, 0.2)'}`
            : `0 2px 8px ${theme.palette.mode === 'dark' ? 'rgba(244, 67, 54, 0.3)' : 'rgba(244, 67, 54, 0.2)'}`
          : 'none',
        '&:hover': {
          borderColor: (theme) => `${isSuccess ? theme.palette.success.main : theme.palette.error.main} !important`,
          boxShadow: (theme) => isSuccess
            ? `0 2px 8px ${theme.palette.mode === 'dark' ? 'rgba(102, 187, 106, 0.2)' : 'rgba(102, 187, 106, 0.1)'}`
            : `0 2px 8px ${theme.palette.mode === 'dark' ? 'rgba(244, 67, 54, 0.2)' : 'rgba(244, 67, 54, 0.1)'}`,
        },
      }}
    >
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        {/* Header with icon */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
          {isSuccess ? (
            <CheckIcon sx={{ fontSize: 20, color: 'success.main' }} />
          ) : (
            <CloseIcon sx={{ fontSize: 20, color: 'error.main' }} />
          )}
          <Typography
            sx={{
              fontSize: '0.85rem',
              fontWeight: 700,
              color: isSuccess ? 'success.main' : 'error.main',
            }}
          >
            {isSuccess ? 'Task completed' : 'Task failed'}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};
