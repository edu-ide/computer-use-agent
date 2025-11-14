import { useAgentStore } from '@/stores/agentStore';
import { FinalStep } from '@/types/agent';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import StopCircleIcon from '@mui/icons-material/StopCircle';
import { Box, Card, CardContent, Typography } from '@mui/material';
import React from 'react';

interface FinalStepCardProps {
  finalStep: FinalStep;
  isActive?: boolean;
}

export const FinalStepCard: React.FC<FinalStepCardProps> = ({ finalStep, isActive = false }) => {
  const setSelectedStepIndex = useAgentStore((state) => state.setSelectedStepIndex);

  const getStatusConfig = () => {
    switch (finalStep.type) {
      case 'success':
        return {
          icon: <CheckIcon sx={{ fontSize: 20, color: 'success.main' }} />,
          label: 'Task completed',
          color: 'success',
        };
      case 'stopped':
        return {
          icon: <StopCircleIcon sx={{ fontSize: 20, color: 'warning.main' }} />,
          label: 'Task stopped',
          color: 'warning',
        };
      case 'max_steps_reached':
        return {
          icon: <HourglassEmptyIcon sx={{ fontSize: 20, color: 'warning.main' }} />,
          label: 'Max steps reached',
          color: 'warning',
        };
      case 'sandbox_timeout':
        return {
          icon: <AccessTimeIcon sx={{ fontSize: 20, color: 'error.main' }} />,
          label: 'Sandbox timeout',
          color: 'error',
        };
      case 'failure':
      default:
        return {
          icon: <CloseIcon sx={{ fontSize: 20, color: 'error.main' }} />,
          label: 'Task failed',
          color: 'error',
        };
    }
  };

  const statusConfig = getStatusConfig();

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
          ? theme.palette[statusConfig.color].main
          : theme.palette.divider} !important`,
        borderRadius: 1.5,
        transition: 'all 0.2s ease',
        cursor: 'pointer',
        boxShadow: isActive
          ? (theme) => `0 2px 8px ${theme.palette.mode === 'dark'
            ? `rgba(${statusConfig.color === 'success' ? '102, 187, 106' : statusConfig.color === 'error' ? '244, 67, 54' : '255, 152, 0'}, 0.3)`
            : `rgba(${statusConfig.color === 'success' ? '102, 187, 106' : statusConfig.color === 'error' ? '244, 67, 54' : '255, 152, 0'}, 0.2)`}`
          : 'none',
        '&:hover': {
          borderColor: (theme) => `${theme.palette[statusConfig.color].main} !important`,
          boxShadow: (theme) => `0 2px 8px ${theme.palette.mode === 'dark'
            ? `rgba(${statusConfig.color === 'success' ? '102, 187, 106' : statusConfig.color === 'error' ? '244, 67, 54' : '255, 152, 0'}, 0.2)`
            : `rgba(${statusConfig.color === 'success' ? '102, 187, 106' : statusConfig.color === 'error' ? '244, 67, 54' : '255, 152, 0'}, 0.1)`}`,
        },
      }}
    >
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        {/* Header with icon */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
          {statusConfig.icon}
          <Typography
            sx={{
              fontSize: '0.85rem',
              fontWeight: 700,
              color: `${statusConfig.color}.main`,
            }}
          >
            {statusConfig.label}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};
