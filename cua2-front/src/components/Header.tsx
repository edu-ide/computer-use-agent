import React, { useState, useEffect, useRef } from 'react';
import { AppBar, Toolbar, Box, Typography, Chip, IconButton, CircularProgress, keyframes, Button } from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import LightModeOutlined from '@mui/icons-material/LightModeOutlined';
import DarkModeOutlined from '@mui/icons-material/DarkModeOutlined';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import InputIcon from '@mui/icons-material/Input';
import OutputIcon from '@mui/icons-material/Output';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import FormatListNumberedIcon from '@mui/icons-material/FormatListNumbered';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import StopCircleIcon from '@mui/icons-material/StopCircle';
import { useAgentStore, selectTrace, selectError, selectIsDarkMode, selectMetadata, selectIsConnectingToE2B, selectFinalStep } from '@/stores/agentStore';

interface HeaderProps {
  isAgentProcessing: boolean;
  onBackToHome?: () => void;
}

// Animation for the running task border - smooth oscillation (primary)
const borderPulse = keyframes`
  0%, 100% {
    border-color: rgba(79, 134, 198, 0.5);
    box-shadow: 0 0 0 0 rgba(79, 134, 198, 0.3);
  }
  50% {
    border-color: rgba(79, 134, 198, 1);
    box-shadow: 0 0 8px 2px rgba(79, 134, 198, 0.4);
  }
`;

// Animation for the background glow (primary)
const backgroundPulse = keyframes`
  0%, 100% {
    background-color: rgba(79, 134, 198, 0.08);
  }
  50% {
    background-color: rgba(79, 134, 198, 0.15);
  }
`;

// Animation for token flash - smooth glow effect
const tokenFlash = keyframes`
  0% {
    filter: brightness(1);
    text-shadow: none;
  }
  25% {
    filter: brightness(1.4);
    text-shadow: 0 0 8px rgba(79, 134, 198, 0.6);
  }
  100% {
    filter: brightness(1);
    text-shadow: none;
  }
`;

// Animation for token icon flash
const iconFlash = keyframes`
  0% {
    filter: brightness(1);
    transform: scale(1);
  }
  25% {
    filter: brightness(1.6);
    transform: scale(1.15);
  }
  100% {
    filter: brightness(1);
    transform: scale(1);
  }
`;

export const Header: React.FC<HeaderProps> = ({ isAgentProcessing, onBackToHome }) => {
  const trace = useAgentStore(selectTrace);
  const error = useAgentStore(selectError);
  const finalStep = useAgentStore(selectFinalStep);
  const isDarkMode = useAgentStore(selectIsDarkMode);
  const toggleDarkMode = useAgentStore((state) => state.toggleDarkMode);
  const metadata = useAgentStore(selectMetadata);
  const isConnectingToE2B = useAgentStore(selectIsConnectingToE2B);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [inputTokenFlash, setInputTokenFlash] = useState(false);
  const [outputTokenFlash, setOutputTokenFlash] = useState(false);
  const prevInputTokens = useRef(0);
  const prevOutputTokens = useRef(0);

  // Update elapsed time every 100ms when agent is processing
  useEffect(() => {
    if (isAgentProcessing && trace?.timestamp) {
      const interval = setInterval(() => {
        const now = new Date();
        const startTime = new Date(trace.timestamp);
        const elapsed = (now.getTime() - startTime.getTime()) / 1000;
        setElapsedTime(elapsed);
      }, 100);

      return () => clearInterval(interval);
    } else if (metadata && metadata.duration > 0) {
      setElapsedTime(metadata.duration);
    }
  }, [isAgentProcessing, trace?.timestamp, metadata]);

  // Detect token changes and trigger flash animation
  useEffect(() => {
    if (metadata) {
      // Input tokens changed
      if (metadata.inputTokensUsed > prevInputTokens.current && prevInputTokens.current > 0) {
        setInputTokenFlash(true);
        setTimeout(() => setInputTokenFlash(false), 800);
      }
      prevInputTokens.current = metadata.inputTokensUsed;

      // Output tokens changed
      if (metadata.outputTokensUsed > prevOutputTokens.current && prevOutputTokens.current > 0) {
        setOutputTokenFlash(true);
        setTimeout(() => setOutputTokenFlash(false), 800);
      }
      prevOutputTokens.current = metadata.outputTokensUsed;
    }
  }, [metadata?.inputTokensUsed, metadata?.outputTokensUsed]);

  // Determine task status - Use finalStep as source of truth
  const getTaskStatus = () => {
    // If we have a final step, use its type
    if (finalStep) {
      switch (finalStep.type) {
        case 'failure':
          return { label: 'Task failed', color: 'error', icon: <CloseIcon sx={{ fontSize: 16, color: 'error.main' }} /> };
        case 'stopped':
          return { label: 'Task stopped', color: 'warning', icon: <StopCircleIcon sx={{ fontSize: 16, color: 'warning.main' }} /> };
        case 'max_steps_reached':
          return { label: 'Max steps reached', color: 'warning', icon: <HourglassEmptyIcon sx={{ fontSize: 16, color: 'warning.main' }} /> };
        case 'success':
          return { label: 'Completed', color: 'success', icon: <CheckIcon sx={{ fontSize: 16, color: 'success.main' }} /> };
      }
    }
    // Otherwise check running states
    if (isConnectingToE2B) return { label: 'Connecting to E2B...', color: 'primary', icon: <CircularProgress size={16} thickness={5} sx={{ color: 'primary.main' }} /> };
    if (isAgentProcessing || trace?.isRunning) return { label: 'Running', color: 'primary', icon: <CircularProgress size={16} thickness={5} sx={{ color: 'primary.main' }} /> };
    return { label: 'Ready', color: 'default', icon: <CheckIcon sx={{ fontSize: 16, color: 'text.secondary' }} /> };
  };

  const taskStatus = getTaskStatus();

  // Extract model name from modelId (e.g., "Qwen/Qwen3-VL-8B-Instruct" -> "Qwen3-VL-8B-Instruct")
  const modelName = trace?.modelId?.split('/').pop() || 'Unknown Model';

  // Handler for emergency stop
  const handleEmergencyStop = () => {
    const stopTask = (window as Window & { __stopCurrentTask?: () => void }).__stopCurrentTask;
    if (stopTask) {
      stopTask();
    }
  };

  return (
    <AppBar
      position="static"
      elevation={0}
      sx={{
        backgroundColor: 'background.paper',
        borderBottom: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Toolbar disableGutters sx={{ px: 2, py: 2.5, flexDirection: 'column', alignItems: 'stretch', gap: 0 }}>
        {/* First row: Back button + Task info + Connection Status */}
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', gap: 3 }}>
          {/* Left side: Back button + Task info */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flex: 1, minWidth: 0 }}>
            <IconButton
              onClick={onBackToHome}
              size="small"
              sx={{
                color: 'primary.main',
                backgroundColor: 'primary.50',
                border: '1px solid',
                borderColor: 'primary.200',
                cursor: 'pointer',
                '&:hover': {
                  backgroundColor: 'primary.100',
                  borderColor: 'primary.main',
                },
              }}
            >
              <ArrowBackIcon fontSize="small" />
            </IconButton>
            <Typography
              variant="body2"
              sx={{
                color: 'text.primary',
                fontWeight: 700,
                fontSize: '1rem',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {trace?.instruction || 'No task running'}
            </Typography>
          </Box>

          {/* Right side: Emergency Stop + Dark Mode */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {/* Emergency Stop Button - Only show when agent is processing */}
            {isAgentProcessing && (
              <Button
                onClick={handleEmergencyStop}
                variant="outlined"
                size="small"
                startIcon={<StopCircleIcon />}
                sx={{
                  color: 'error.main',
                  borderColor: 'error.main',
                  backgroundColor: 'transparent',
                  fontWeight: 600,
                  fontSize: '0.8rem',
                  px: 1.5,
                  py: 0.5,
                  textTransform: 'none',
                  '&:hover': {
                    backgroundColor: 'error.50',
                    borderColor: 'error.dark',
                  },
                }}
              >
                Stop
              </Button>
            )}

            <IconButton
              onClick={toggleDarkMode}
              size="small"
              sx={{
                color: 'primary.main',
                backgroundColor: 'primary.50',
                border: '1px solid',
                borderColor: 'primary.200',
                '&:hover': {
                  backgroundColor: 'primary.100',
                  borderColor: 'primary.main',
                },
              }}
            >
              {isDarkMode ? <LightModeOutlined fontSize="small" /> : <DarkModeOutlined fontSize="small" />}
            </IconButton>
          </Box>
        </Box>

        {/* Second row: Status + Model + Metadata - Only show when we have trace data */}
        {trace && (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1.5,
              pl: 5.5,
              pr: 1,
              pt: .5,
              mt: .5,
            }}
          >
            {/* Status Badge - Compact */}
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                px: 1,
                py: 0.25,
                borderRadius: 1,
                backgroundColor:
                  taskStatus.color === 'primary' ? 'primary.50' :
                  taskStatus.color === 'success' ? 'success.50' :
                  taskStatus.color === 'error' ? 'error.50' :
                  taskStatus.color === 'warning' ? 'warning.50' :
                  'action.hover',
                border: '1px solid',
                borderColor:
                  taskStatus.color === 'primary' ? 'primary.main' :
                  taskStatus.color === 'success' ? 'success.main' :
                  taskStatus.color === 'error' ? 'error.main' :
                  taskStatus.color === 'warning' ? 'warning.main' :
                  'divider',
              }}
            >
              {taskStatus.icon}
              <Typography
                variant="caption"
                sx={{
                  fontSize: '0.7rem',
                  fontWeight: 700,
                  color:
                    taskStatus.color === 'primary' ? 'primary.main' :
                    taskStatus.color === 'success' ? 'success.main' :
                    taskStatus.color === 'error' ? 'error.main' :
                    taskStatus.color === 'warning' ? 'warning.main' :
                    'text.primary',
                }}
              >
                {taskStatus.label}
              </Typography>
            </Box>

            {/* Divider */}
            <Box sx={{ width: '1px', height: 16, backgroundColor: 'divider' }} />

            {/* Model */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <SmartToyIcon sx={{ fontSize: '0.85rem', color: 'primary.main' }} />
              <Typography
                variant="caption"
                sx={{
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  color: 'text.primary',
                }}
              >
                {modelName}
              </Typography>
            </Box>

            {/* Steps Count */}
            {metadata && (
              <>
                <Box sx={{ width: '1px', height: 16, backgroundColor: 'divider' }} />
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Typography
                    variant="caption"
                    sx={{
                      fontSize: '0.75rem',
                      fontWeight: 700,
                      color: 'text.primary',
                      mr: 0.5,
                    }}
                  >
                    {metadata.numberOfSteps}
                  </Typography>
                  <Typography
                    variant="caption"
                    sx={{
                      fontSize: '0.7rem',
                      fontWeight: 400,
                      color: 'text.secondary',
                    }}
                  >
                    {metadata.numberOfSteps === 1 ? 'Step' : 'Steps'}
                  </Typography>
                </Box>
              </>
            )}

            {/* Time */}
            {(isAgentProcessing || metadata) && (
              <>
                <Box sx={{ width: '1px', height: 16, backgroundColor: 'divider' }} />
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <AccessTimeIcon sx={{ fontSize: '0.85rem', color: 'primary.main' }} />
                  <Typography
                    variant="caption"
                    sx={{
                      fontSize: '0.75rem',
                      fontWeight: 700,
                      color: 'text.primary',
                      minWidth: '45px',
                      textAlign: 'left',
                    }}
                  >
                    {elapsedTime.toFixed(1)}s
                  </Typography>
                </Box>
              </>
            )}

            {/* Input Tokens */}
            {metadata && metadata.inputTokensUsed > 0 && (
              <>
                <Box sx={{ width: '1px', height: 16, backgroundColor: 'divider' }} />
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <InputIcon
                    sx={{
                      fontSize: '0.85rem',
                      color: 'primary.main',
                      transition: 'all 0.2s ease',
                      animation: inputTokenFlash ? `${iconFlash} 0.8s ease-out` : 'none',
                    }}
                  />
                  <Box
                    sx={{
                      transition: 'all 0.2s ease',
                      animation: inputTokenFlash ? `${tokenFlash} 0.8s ease-out` : 'none',
                    }}
                  >
                    <Typography
                      variant="caption"
                      sx={{
                        fontSize: '0.75rem',
                        fontWeight: 700,
                        color: 'text.primary',
                      }}
                    >
                      {metadata.inputTokensUsed.toLocaleString()}
                    </Typography>
                  </Box>
                </Box>
              </>
            )}

            {/* Output Tokens */}
            {metadata && metadata.outputTokensUsed > 0 && (
              <>
                <Box sx={{ width: '1px', height: 16, backgroundColor: 'divider' }} />
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <OutputIcon
                    sx={{
                      fontSize: '0.85rem',
                      color: 'primary.main',
                      transition: 'all 0.2s ease',
                      animation: outputTokenFlash ? `${iconFlash} 0.8s ease-out` : 'none',
                    }}
                  />
                  <Box
                    sx={{
                      transition: 'all 0.2s ease',
                      animation: outputTokenFlash ? `${tokenFlash} 0.8s ease-out` : 'none',
                    }}
                  >
                    <Typography
                      variant="caption"
                      sx={{
                        fontSize: '0.75rem',
                        fontWeight: 700,
                        color: 'text.primary',
                      }}
                    >
                      {metadata.outputTokensUsed.toLocaleString()}
                    </Typography>
                  </Box>
                </Box>
              </>
            )}
          </Box>
        )}
      </Toolbar>
    </AppBar>
  );
};
