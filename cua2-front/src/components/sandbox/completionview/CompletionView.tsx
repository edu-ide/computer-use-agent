import React from 'react';
import { Box, Typography, Button, Divider, Alert, Paper } from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import StopCircleIcon from '@mui/icons-material/StopCircle';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import AddIcon from '@mui/icons-material/Add';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import AssignmentIcon from '@mui/icons-material/Assignment';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import InputIcon from '@mui/icons-material/Input';
import OutputIcon from '@mui/icons-material/Output';
import FormatListNumberedIcon from '@mui/icons-material/FormatListNumbered';
import { FinalStep, AgentTrace, AgentStep } from '@/types/agent';
import { DownloadGifButton } from './DownloadGifButton';
import { DownloadJsonButton } from './DownloadJsonButton';

interface CompletionViewProps {
  finalStep: FinalStep;
  trace?: AgentTrace;
  steps?: AgentStep[];
  finalAnswer?: string | null;
  isGenerating: boolean;
  gifError: string | null;
  onGenerateGif: () => void;
  onDownloadJson: () => void;
  onBackToHome: () => void;
}

/**
 * Component displaying the completion status (success or failure) of a task
 */
export const CompletionView: React.FC<CompletionViewProps> = ({
  finalStep,
  trace,
  steps,
  finalAnswer,
  isGenerating,
  gifError,
  onGenerateGif,
  onDownloadJson,
  onBackToHome,
}) => {
  const getStatusConfig = () => {
    switch (finalStep.type) {
      case 'success':
        return {
          icon: <CheckIcon sx={{ fontSize: 28 }} />,
          title: 'Task Completed Successfully!',
          color: 'success.main',
        };
      case 'stopped':
        return {
          icon: <StopCircleIcon sx={{ fontSize: 28 }} />,
          title: 'Task Stopped',
          color: 'warning.main',
        };
      case 'max_steps_reached':
        return {
          icon: <HourglassEmptyIcon sx={{ fontSize: 28 }} />,
          title: 'Maximum Steps Reached',
          color: 'warning.main',
        };
      case 'sandbox_timeout':
        return {
          icon: <AccessTimeIcon sx={{ fontSize: 28 }} />,
          title: 'Sandbox Timeout',
          color: 'error.main',
        };
      case 'failure':
      default:
        return {
          icon: <CloseIcon sx={{ fontSize: 28 }} />,
          title: 'Task Failed',
          color: 'error.main',
        };
    }
  };

  const statusConfig = getStatusConfig();

  // Format model name for display
  const formatModelName = (modelId: string) => {
    const parts = modelId.split('/');
    return parts.length > 1 ? parts[1] : modelId;
  };

  return (
    <Box
      sx={{
        width: '100%',
        maxWidth: 600,
        mx: 'auto',
        p: 2,
        display: 'flex',
        flexDirection: 'column',
        gap: 1.5,
      }}
    >
      {/* Status Header - Compact */}
      <Box sx={{ textAlign: 'center', mb: 0.5 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1.5, mb: 0.75 }}>
          <Box
            sx={{
              width: 40,
              height: 40,
              borderRadius: '50%',
              backgroundColor: statusConfig.color,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: (theme) => {
                const rgba = finalStep.type === 'success'
                  ? '102, 187, 106'
                  : (finalStep.type === 'failure' || finalStep.type === 'sandbox_timeout')
                    ? '244, 67, 54'
                    : '255, 152, 0';
                return `0 2px 8px ${theme.palette.mode === 'dark' ? `rgba(${rgba}, 0.3)` : `rgba(${rgba}, 0.2)`}`;
              },
            }}
          >
            {React.cloneElement(statusConfig.icon, { sx: { fontSize: 24, color: 'white' } })}
          </Box>
          <Typography
            variant="h6"
            sx={{
              fontWeight: 700,
              color: statusConfig.color,
              fontSize: '1.1rem',
              letterSpacing: '-0.5px',
            }}
          >
            {statusConfig.title}
          </Typography>
        </Box>
      </Box>

      {/* Single Report Box - Task + Agent + Response + Metrics */}
      <Paper
        elevation={0}
        sx={{
          p: 2.5,
          backgroundColor: (theme) => theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.03)',
          borderRadius: 1.5,
          border: '1px solid',
          borderColor: 'divider',
        }}
      >
        {/* Task */}
        {trace?.instruction && (
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.5 }}>
              <AssignmentIcon sx={{ fontSize: 18, color: 'text.secondary', mt: 0.25, flexShrink: 0 }} />
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Typography
                  variant="caption"
                  sx={{
                    fontWeight: 700,
                    color: 'text.secondary',
                    fontSize: '0.7rem',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    display: 'block',
                    mb: 0.5,
                  }}
                >
                  Task
                </Typography>
                <Typography
                  variant="body2"
                  sx={{
                    color: 'text.primary',
                    fontWeight: 700,
                    lineHeight: 1.5,
                    fontSize: '0.85rem',
                  }}
                >
                  {trace.instruction}
                </Typography>
              </Box>
            </Box>
          </Box>
        )}

        {/* Agent Response */}
        {finalAnswer && (
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.5 }}>
              <ChatBubbleOutlineIcon
                sx={{
                  fontSize: 18,
                  color: 'text.secondary',
                  mt: 0.25,
                  flexShrink: 0
                }}
              />
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Typography
                  variant="caption"
                  sx={{
                    fontWeight: 700,
                    color: 'text.secondary',
                    fontSize: '0.7rem',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    display: 'block',
                    mb: 0.75,
                  }}
                >
                  Agent Response
                </Typography>
                <Typography
                  variant="body2"
                  sx={{
                    color: 'text.primary',
                    lineHeight: 1.5,
                    fontSize: '0.85rem',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                  }}
                >
                  {finalAnswer}
                </Typography>
              </Box>
            </Box>
          </Box>
        )}

        {/* Divider before metrics */}
        <Divider sx={{ my: 2 }} />

        {/* Metrics */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1.5,
            flexWrap: 'wrap',
            justifyContent: 'center',
          }}
        >
          {/* Agent */}
          {trace?.modelId && (
            <>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <SmartToyIcon sx={{ fontSize: '0.85rem', color: 'primary.main' }} />
                <Typography
                  variant="caption"
                  sx={{
                    color: 'text.primary',
                    fontFamily: 'monospace',
                    fontSize: '0.75rem',
                    fontWeight: 700,
                  }}
                >
                  {formatModelName(trace.modelId)}
                </Typography>
              </Box>

              {/* Divider */}
              <Box sx={{ width: '1px', height: 16, backgroundColor: 'divider' }} />
            </>
          )}

          {/* Steps Count */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <FormatListNumberedIcon sx={{ fontSize: '0.85rem', color: 'primary.main' }} />
            <Typography
              variant="caption"
              sx={{
                fontSize: '0.75rem',
                fontWeight: 700,
                color: 'text.primary',
                mr: 0.5,
              }}
            >
              {finalStep.metadata.numberOfSteps}
            </Typography>
            <Typography
              variant="caption"
              sx={{
                fontSize: '0.7rem',
                fontWeight: 400,
                color: 'text.secondary',
              }}
            >
              {finalStep.metadata.numberOfSteps === 1 ? 'Step' : 'Steps'}
            </Typography>
          </Box>

          {/* Divider */}
          <Box sx={{ width: '1px', height: 16, backgroundColor: 'divider' }} />

          {/* Duration */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <AccessTimeIcon sx={{ fontSize: '0.85rem', color: 'primary.main' }} />
            <Typography
              variant="caption"
              sx={{
                fontSize: '0.75rem',
                fontWeight: 700,
                color: 'text.primary',
              }}
            >
              {finalStep.metadata.duration.toFixed(1)}s
            </Typography>
          </Box>

          {/* Divider */}
          <Box sx={{ width: '1px', height: 16, backgroundColor: 'divider' }} />

          {/* Input Tokens */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <InputIcon sx={{ fontSize: '0.85rem', color: 'primary.main' }} />
            <Typography
              variant="caption"
              sx={{
                fontSize: '0.75rem',
                fontWeight: 700,
                color: 'text.primary',
              }}
            >
              {finalStep.metadata.inputTokensUsed.toLocaleString()}
            </Typography>
          </Box>

          {/* Divider */}
          <Box sx={{ width: '1px', height: 16, backgroundColor: 'divider' }} />

          {/* Output Tokens */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <OutputIcon sx={{ fontSize: '0.85rem', color: 'primary.main' }} />
            <Typography
              variant="caption"
              sx={{
                fontSize: '0.75rem',
                fontWeight: 700,
                color: 'text.primary',
              }}
            >
              {finalStep.metadata.outputTokensUsed.toLocaleString()}
            </Typography>
          </Box>
        </Box>
      </Paper>

      {/* GIF Error Alert */}
      {gifError && (
        <Alert severity="error" sx={{ fontSize: '0.72rem', py: 0.5 }}>
          {gifError}
        </Alert>
      )}

      {/* Action Buttons */}
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          gap: 1.5,
          alignItems: 'center',
        }}
      >
        {/* Download buttons */}
        <Box
          sx={{
            display: 'flex',
            gap: 1,
            justifyContent: 'center',
            flexWrap: 'wrap',
          }}
        >
          <DownloadGifButton
            isGenerating={isGenerating}
            onClick={onGenerateGif}
            disabled={!steps || steps.length === 0}
          />
          <DownloadJsonButton onClick={onDownloadJson} disabled={!trace} />
        </Box>

        {/* New Task button - larger and below */}
        <Button
          variant="contained"
          startIcon={<AddIcon sx={{ fontSize: 20 }} />}
          onClick={onBackToHome}
          color="primary"
          sx={{
            textTransform: 'none',
            fontWeight: 700,
            fontSize: '0.9rem',
            px: 3,
            py: 1,
            boxShadow: 2,
            minWidth: 200,
            '&:hover': {
              boxShadow: 4,
            },
          }}
        >
          New Task
        </Button>
      </Box>
    </Box>
  );
};
