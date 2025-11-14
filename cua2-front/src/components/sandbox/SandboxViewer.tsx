import { useGifGenerator } from '@/hooks/useGifGenerator';
import { useJsonExporter } from '@/hooks/useJsonExporter';
import { selectError, selectFinalStep, selectSteps, selectTrace, useAgentStore } from '@/stores/agentStore';
import { AgentStep, AgentTraceMetadata } from '@/types/agent';
import ImageIcon from '@mui/icons-material/Image';
import MonitorIcon from '@mui/icons-material/Monitor';
import PlayCircleIcon from '@mui/icons-material/PlayCircle';
import { Box, Button, CircularProgress, keyframes, Typography } from '@mui/material';
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { CompletionView } from './completionview/CompletionView';

// Animation for live indicator
const livePulse = keyframes`
  0%, 100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.7;
    transform: scale(1.2);
  }
`;

interface SandboxViewerProps {
  vncUrl: string;
  isAgentProcessing?: boolean;
  metadata?: AgentTraceMetadata;
  traceStartTime?: Date;
  selectedStep?: AgentStep | null; // The step to display in time-travel mode
  isRunning?: boolean; // Is the agent currently running
}

export const SandboxViewer: React.FC<SandboxViewerProps> = ({
  vncUrl,
  isAgentProcessing = false,
  metadata,
  traceStartTime,
  selectedStep,
  isRunning = false
}) => {
  const navigate = useNavigate();
  const error = useAgentStore(selectError);
  const finalStep = useAgentStore(selectFinalStep);
  const steps = useAgentStore(selectSteps);
  const trace = useAgentStore(selectTrace);
  const resetAgent = useAgentStore((state) => state.resetAgent);
  const setSelectedStepIndex = useAgentStore((state) => state.setSelectedStepIndex);

  // Hook to generate GIF
  const { isGenerating, error: gifError, generateAndDownloadGif } = useGifGenerator({
    steps: steps || [],
    traceId: finalStep?.metadata.traceId || '',
  });

  // Hook to export JSON
  const { downloadTraceAsJson } = useJsonExporter({
    trace,
    steps: steps || [],
    metadata: finalStep?.metadata || metadata,
    finalStep,
  });

  // Extract final_answer from the last step, or fallback to last thought
  const getFinalAnswer = (): string | null => {
    console.log('🔍 getFinalAnswer - steps:', steps);
    if (!steps || steps.length === 0) {
      console.log('❌ No steps available');
      return null;
    }

    // Try to find final_answer in any step (iterate backwards)
    for (let i = steps.length - 1; i >= 0; i--) {
      const step = steps[i];

      if (step.actions && Array.isArray(step.actions)) {
        const finalAnswerAction = step.actions.find(
          (action) => action.function_name === 'final_answer'
        );

        if (finalAnswerAction) {
          // Handle both named parameter and positional argument
          const result = finalAnswerAction?.parameters?.answer || finalAnswerAction?.parameters?.arg_0 || null;
          console.log('✅ Final answer found in step', i + 1, ':', result);
          return result;
        }
      }
    }

    console.log('🔍 No final_answer found, looking for last thought...');

    // Fallback: find the last step with a thought (iterate backwards)
    for (let i = steps.length - 1; i >= 0; i--) {
      const step = steps[i];
      if (step.thought) {
        console.log('📝 Using thought from step', i + 1, 'as fallback:', step.thought);
        return step.thought;
      }
    }

    console.log('❌ No final answer or thought found in any step');
    return null;
  };

  const finalAnswer = getFinalAnswer();
  console.log('🎯 Final answer to display:', finalAnswer);

  // Determine if we should show success/fail status
  const showStatus = !isRunning && !selectedStep && finalStep;

  // Handler to go back to home
  const handleBackToHome = () => {
    // Reset frontend state
    useAgentStore.getState().resetAgent();

    // Reload the page to reconnect websocket
    window.location.href = '/';
  };

  // Handler to go back to live mode
  const handleGoLive = () => {
    setSelectedStepIndex(null);
  };

  return (
    <Box
      sx={{
        flex: '1 1 auto',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        border: '1px solid',
        borderColor: showStatus
          ? ((finalStep?.type === 'failure' || finalStep?.type === 'sandbox_timeout') ? 'error.main' : 'success.main')
          : ((vncUrl || isAgentProcessing) && !selectedStep && !showStatus ? 'primary.main' : 'divider'),
        borderRadius: '12px',
        backgroundColor: 'background.paper',
        transition: 'border 0.3s ease',
        overflow: 'hidden',
      }}
    >
      {/* Live Badge or Go Live Button */}
      {vncUrl && !showStatus && (
        <>
          {!selectedStep ? (
            // Live Badge when in live mode
            <Box
              sx={{
                position: 'absolute',
                top: 12,
                right: 12,
                zIndex: 10,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                px: 2,
                py: 1,
                backgroundColor: (theme) =>
                  theme.palette.mode === 'dark'
                    ? 'rgba(0, 0, 0, 0.7)'
                    : 'rgba(255, 255, 255, 0.9)',
                backdropFilter: 'blur(8px)',
                borderRadius: 0.75,
                border: '1px solid',
                borderColor: 'primary.main',
                boxShadow: (theme) =>
                  theme.palette.mode === 'dark'
                    ? '0 2px 8px rgba(0, 0, 0, 0.4)'
                    : '0 2px 8px rgba(0, 0, 0, 0.1)',
              }}
            >
              <Box
                sx={{
                  width: 10,
                  height: 10,
                  borderRadius: '50%',
                  backgroundColor: 'error.main',
                  animation: `${livePulse} 2s ease-in-out infinite`,
                }}
              />
              <Typography
                variant="caption"
                sx={{
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  color: 'text.primary',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                }}
              >
                Live
              </Typography>
            </Box>
          ) : (
            // Go Live Button when viewing a specific step
            <Button
              onClick={handleGoLive}
              startIcon={<PlayCircleIcon sx={{ fontSize: 20 }} />}
              sx={{
                position: 'absolute',
                top: 12,
                right: 12,
                zIndex: 10,
                px: 2,
                py: 1,
                backgroundColor: (theme) =>
                  theme.palette.mode === 'dark'
                    ? 'rgba(0, 0, 0, 0.7)'
                    : 'rgba(255, 255, 255, 0.9)',
                backdropFilter: 'blur(8px)',
                borderRadius: 0.75,
                border: '1px solid',
                borderColor: 'primary.main',
                boxShadow: (theme) =>
                  theme.palette.mode === 'dark'
                    ? '0 2px 8px rgba(0, 0, 0, 0.4)'
                    : '0 2px 8px rgba(0, 0, 0, 0.1)',
                fontSize: '0.8rem',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
                color: 'primary.main',
                '&:hover': {
                  backgroundColor: (theme) =>
                    theme.palette.mode === 'dark'
                      ? 'rgba(0, 0, 0, 0.85)'
                      : 'rgba(255, 255, 255, 1)',
                  borderColor: 'primary.dark',
                },
              }}
            >
              Go Live
            </Button>
          )}
        </>
      )}

      <Box
        sx={{
          flex: 1,
          minHeight: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {showStatus && finalStep ? (
          // Show success/fail status when agent has completed
          <CompletionView
            finalStep={finalStep}
            trace={trace}
            steps={steps}
            finalAnswer={finalAnswer}
            isGenerating={isGenerating}
            gifError={gifError}
            onGenerateGif={generateAndDownloadGif}
            onDownloadJson={downloadTraceAsJson}
            onBackToHome={handleBackToHome}
          />
        ) : selectedStep ? (
          // Time-travel mode: Show screenshot of selected step
          <Box
            sx={{
              width: '100%',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              overflow: 'auto',
              backgroundColor: 'black',
              position: 'relative',
            }}
          >
            {selectedStep.image ? (
              <img
                src={selectedStep.image}
                alt="Step screenshot"
                style={{
                  maxWidth: '100%',
                  maxHeight: '100%',
                  objectFit: 'contain',
                }}
              />
            ) : (
              <Box
                sx={{
                  textAlign: 'center',
                  p: 4,
                  color: 'text.secondary',
                  width: '100%',
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <ImageIcon sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
                <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5, fontSize: '0.875rem', color: 'text.primary' }}>
                  No screenshot available
                </Typography>
                <Typography variant="caption" sx={{ fontSize: '0.75rem', color: 'text.secondary' }}>
                  This step doesn't have a screenshot
                </Typography>
              </Box>
            )}
          </Box>
        ) : vncUrl ? (
          // Live mode: Show VNC stream
          <iframe
            src={vncUrl}
            style={{ width: '100%', height: '100%', border: 'none' }}
            title="OS Stream"
            lang="en"
          />
        ) : isAgentProcessing ? (
          // Loading state
          <Box
            sx={{
              textAlign: 'center',
              p: 4,
              color: 'text.secondary',
              width: '100%',
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <CircularProgress
              size={48}
              sx={{
                mb: 2,
                color: 'primary.main'
              }}
            />
            <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5, fontSize: '0.875rem', color: 'text.primary' }}>
              Connecting to E2B...
            </Typography>
            <Typography variant="caption" sx={{ fontSize: '0.75rem', color: 'text.secondary' }}>
              Setting up sandbox environment
            </Typography>
          </Box>
        ) : (
          // No stream available
          <Box
            sx={{
              textAlign: 'center',
              p: 4,
              color: 'text.secondary',
              width: '100%',
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <MonitorIcon sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
            <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5, fontSize: '0.875rem' }}>
              No stream available
            </Typography>
            <Typography variant="caption" sx={{ fontSize: '0.75rem', color: 'text.secondary' }}>
              Stream will appear when agent starts
            </Typography>
          </Box>
        )}
      </Box>
    </Box>
  );
};
