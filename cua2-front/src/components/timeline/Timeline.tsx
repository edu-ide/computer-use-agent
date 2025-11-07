import React, { useRef, useEffect } from 'react';
import { Box, Typography, CircularProgress, Button } from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import StopCircleIcon from '@mui/icons-material/StopCircle';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import CableIcon from '@mui/icons-material/Cable';
import { AgentTraceMetadata } from '@/types/agent';
import { useAgentStore, selectSelectedStepIndex, selectFinalStep, selectIsConnectingToE2B, selectIsAgentProcessing } from '@/stores/agentStore';

interface TimelineProps {
  metadata: AgentTraceMetadata;
  isRunning: boolean;
}

export const Timeline: React.FC<TimelineProps> = ({ metadata, isRunning }) => {
  const timelineRef = useRef<HTMLDivElement>(null);
  const selectedStepIndex = useAgentStore(selectSelectedStepIndex);
  const setSelectedStepIndex = useAgentStore((state) => state.setSelectedStepIndex);
  const finalStep = useAgentStore(selectFinalStep);
  const isConnectingToE2B = useAgentStore(selectIsConnectingToE2B);
  const isAgentProcessing = useAgentStore(selectIsAgentProcessing);

  // Show connection indicator if connecting or if we have started processing
  const showConnectionIndicator = isConnectingToE2B || isAgentProcessing || (metadata.numberOfSteps > 0) || finalStep;

  // Generate array of steps with their status
  // Only show completed steps + current step if running
  const totalStepsToShow = isRunning && !isConnectingToE2B
    ? metadata.numberOfSteps + 1  // Show completed steps + current step
    : metadata.numberOfSteps;     // Show only completed steps when not running

  // Calculate total width for the line (including finalStep if present)
  const lineWidth = finalStep
    ? `calc(${totalStepsToShow} * (40px + 12px) + 52px)` // Add space for finalStep (40px + 12px gap)
    : `calc(${totalStepsToShow} * (40px + 12px))`;

  const steps = Array.from({ length: totalStepsToShow }, (_, index) => ({
    stepNumber: index + 1,
    stepIndex: index,
    isCompleted: index < metadata.numberOfSteps,
    // Step is current if: we're at the right index AND running AND not connecting to E2B
    isCurrent: (index === metadata.numberOfSteps && isRunning && !isConnectingToE2B) ||
               (index === 0 && metadata.numberOfSteps === 0 && isRunning && !isConnectingToE2B),
    isSelected: selectedStepIndex === index,
  }));

  // Handle step click
  const handleStepClick = (stepIndex: number, isCompleted: boolean, isCurrent: boolean) => {
    if (isCompleted) {
      setSelectedStepIndex(stepIndex);
    } else if (isCurrent) {
      // Clicking on the current step (with animation) goes back to live mode
      setSelectedStepIndex(null);
    }
  };

  // Handle final step click (goes to live mode showing the final status)
  const handleFinalStepClick = () => {
    setSelectedStepIndex(null);
  };

  // Auto-scroll to current step while running
  useEffect(() => {
    if (timelineRef.current && isRunning) {
      // Only auto-scroll while running, not when finished
      const currentStepElement = timelineRef.current.querySelector(`[data-step="${metadata.numberOfSteps}"]`);
      if (currentStepElement) {
        currentStepElement.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
      }
    }
  }, [metadata.numberOfSteps, isRunning]);

  return (
    <Box
      sx={{
        p: 2,
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: '12px',
        backgroundColor: 'background.paper',
        flexShrink: 0,
      }}
    >
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
        {/* Header with step count */}
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="h6" sx={{ fontSize: '0.9rem', fontWeight: 700, color: 'text.primary' }}>
            Timeline
            {selectedStepIndex !== null && (
              <Typography component="span" sx={{ ml: 1, color: 'text.secondary', fontWeight: 500, fontSize: '0.65rem' }}>
                - Viewing step {selectedStepIndex + 1}
              </Typography>
            )}
          </Typography>
          {selectedStepIndex !== null && (
            <Button
              size="small"
              variant="outlined"
              onClick={handleFinalStepClick}
              sx={{
                textTransform: 'none',
                fontSize: '0.7rem',
                fontWeight: 600,
                px: 1.5,
                py: 0.25,
                minWidth: 'auto',
                color: 'text.secondary',
                borderColor: 'divider',
                '&:hover': {
                  backgroundColor: (theme) => theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.03)',
                  borderColor: 'text.secondary',
                },
              }}
            >
              Back to latest step
            </Button>
          )}
        </Box>

        {/* Horizontal scrollable step indicators */}
        <Box
          ref={timelineRef}
          sx={{
            display: 'flex',
            alignItems: 'center',
            overflowX: 'auto',
            overflowY: 'hidden',
            gap: 1.5,
            py: 1.5,
            height: 60,
            position: 'relative',
            // Hide scrollbar completely
            scrollbarWidth: 'none', // Firefox
            '&::-webkit-scrollbar': {
              display: 'none', // Chrome, Safari, Edge
            },
            // Horizontal line crossing through circles
            '&::before': {
              content: '""',
              position: 'absolute',
              left: "25px",
              // Calculate width to cover visible steps + finalStep if present
              width: lineWidth,
              top: '19.5px',
              transform: 'translateY(-50%)',
              transition: 'width 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
              height: '2px',
              backgroundColor: (theme) => theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.3)',
              zIndex: 0,
              pointerEvents: 'none',
            },
          }}
        >
          {/* Connection indicator (step 0) */}
          {showConnectionIndicator && (
            <Box
              data-step="connection"
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 0.75,
                minWidth: 40,
                flexShrink: 0,
                position: 'relative',
                zIndex: 1,
              }}
            >
              {/* White circle background to hide the line */}
              <Box
                sx={{
                  position: 'relative',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: 28,
                  width: 28,
                }}
              >
                {/* White background to hide the line */}
                <Box
                  sx={{
                    position: 'absolute',
                    width: 28,
                    height: 28,
                    borderRadius: '50%',
                    backgroundColor: 'background.paper',
                    zIndex: 0,
                  }}
                />

                {/* Connection icon */}
                {isConnectingToE2B ? (
                  <CircularProgress
                    size={20}
                    thickness={5}
                    sx={{
                      color: 'primary.main',
                      position: 'relative',
                      zIndex: 1,
                    }}
                  />
                ) : (
                  <CableIcon
                    sx={{
                      fontSize: 20,
                      color: 'success.main',
                      position: 'relative',
                      zIndex: 1,
                    }}
                  />
                )}
              </Box>

              {/* Connection label */}
              <Typography
                variant="caption"
                sx={{
                  fontSize: '0.7rem',
                  fontWeight: 700,
                  color: isConnectingToE2B ? 'primary.main' : 'success.main',
                  whiteSpace: 'nowrap',
                }}
              >
                {isConnectingToE2B ? 'Connecting' : 'Connected'}
              </Typography>
            </Box>
          )}

          {/* Render steps and insert final step at the right position */}
          {steps.map((step, index) => (
            <React.Fragment key={step.stepNumber}>
              <Box
                data-step={step.stepNumber}
                onClick={() => handleStepClick(step.stepIndex, step.isCompleted, step.isCurrent)}
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 0.75,
                  minWidth: 40,
                  flexShrink: 0,
                  position: 'relative',
                  zIndex: 1,
                  cursor: (step.isCompleted || step.isCurrent) ? 'pointer' : 'default',
                  '&:hover': (step.isCompleted || step.isCurrent) ? {
                    '& .step-dot': {
                      transform: 'scale(1.15)',
                    },
                  } : {},
                }}
              >
                {/* White circle background to hide the line */}
                <Box
                  sx={{
                    position: 'relative',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    height: 28,
                    width: 28,
                  }}
                >
                  {/* White background to hide the line */}
                  <Box
                    sx={{
                      position: 'absolute',
                      width: 28,
                      height: 28,
                      borderRadius: '50%',
                      backgroundColor: 'background.paper',
                      zIndex: 0,
                    }}
                  />

                  {/* Step dot */}
                  {step.isCurrent ? (
                    <Box
                      sx={{
                        position: 'relative',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        zIndex: 1,
                      }}
                    >
                      <CircularProgress
                        size={20}
                        thickness={5}
                        sx={{
                          color: 'primary.main',
                          position: 'absolute',
                        }}
                      />
                      <Box
                        sx={{
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          backgroundColor: 'white',
                          position: 'absolute',
                          pointerEvents: 'none',
                          boxShadow: '0 0 4px rgba(0,0,0,0.2)',
                        }}
                      />
                    </Box>
                  ) : (
                    <Box
                      sx={{
                        position: 'relative',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        zIndex: 1,
                      }}
                    >
                      <Box
                        className="step-dot"
                        sx={{
                          width: step.isSelected ? 20 : step.isCompleted ? 14 : 12,
                          height: step.isSelected ? 20 : step.isCompleted ? 14 : 12,
                          borderRadius: '50%',
                          // Always keep steps in primary color (blue)
                          backgroundColor: step.isCompleted
                            ? 'primary.main' // Blue for completed steps
                            : (theme) => theme.palette.mode === 'dark' ? 'grey.800' : 'grey.300', // Light grey for future steps
                          transition: 'all 0.2s ease',
                          boxShadow: step.isCompleted || step.isSelected
                            ? step.isSelected
                              ? '0 0 8px rgba(255, 167, 38, 0.5)'
                              : '0 2px 4px rgba(0,0,0,0.1)'
                            : 'none',
                        }}
                      />
                      {/* White dot for selected step */}
                      {step.isSelected && (
                        <Box
                          sx={{
                            width: 8,
                            height: 8,
                            borderRadius: '50%',
                            backgroundColor: 'white',
                            position: 'absolute',
                          }}
                        />
                      )}
                    </Box>
                  )}
                </Box>

                {/* Step number - show for all steps */}
                <Typography
                  variant="caption"
                  sx={{
                    fontSize: '0.7rem',
                    fontWeight: step.isSelected || step.isCurrent ? 900 : 400,
                    color: step.isCurrent
                      ? 'primary.main'
                      : (step.isCompleted || step.isSelected
                        ? 'text.primary'
                        : (theme) => theme.palette.mode === 'dark' ? 'grey.700' : 'grey.400'),
                    whiteSpace: 'nowrap',
                    lineHeight: 1,
                  }}
                >
                  {step.stepNumber}
                </Typography>
              </Box>

              {/* Insert final step indicator right after the last completed step */}
              {finalStep && step.stepNumber === metadata.numberOfSteps && (
            <Box
              data-step="final"
              onClick={handleFinalStepClick}
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 0.75,
                minWidth: 40,
                flexShrink: 0,
                position: 'relative',
                zIndex: 1,
                cursor: 'pointer',
                '&:hover': {
                  '& .final-step-icon': {
                    transform: 'scale(1.15)',
                  },
                },
              }}
            >
              {/* White circle background to hide the line */}
              <Box
                sx={{
                  position: 'relative',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: 28,
                  width: 28,
                }}
              >
                {/* White background to hide the line */}
                <Box
                  sx={{
                    position: 'absolute',
                    width: 28,
                    height: 28,
                    borderRadius: '50%',
                    backgroundColor: 'background.paper',
                    zIndex: 0,
                  }}
                />

                {/* Final step icon */}
                <Box
                  className="final-step-icon"
                  sx={{
                    width: selectedStepIndex === null ? 20 : 18,
                    height: selectedStepIndex === null ? 20 : 18,
                    borderRadius: '50%',
                    backgroundColor:
                      finalStep.type === 'success' ? 'success.main' :
                      finalStep.type === 'stopped' || finalStep.type === 'max_steps_reached' ? 'warning.main' :
                      'error.main',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'all 0.2s ease',
                    boxShadow: selectedStepIndex === null
                      ? finalStep.type === 'success'
                        ? '0 2px 8px rgba(102, 187, 106, 0.4)'
                        : finalStep.type === 'stopped' || finalStep.type === 'max_steps_reached'
                          ? '0 2px 8px rgba(255, 152, 0, 0.4)'
                          : '0 2px 8px rgba(244, 67, 54, 0.4)'
                      : '0 2px 4px rgba(0,0,0,0.1)',
                    position: 'relative',
                    zIndex: 1,
                  }}
                >
                  {finalStep.type === 'success' ? (
                    <CheckIcon sx={{ fontSize: 14, color: 'white' }} />
                  ) : finalStep.type === 'stopped' ? (
                    <StopCircleIcon sx={{ fontSize: 14, color: 'white' }} />
                  ) : finalStep.type === 'max_steps_reached' ? (
                    <HourglassEmptyIcon sx={{ fontSize: 14, color: 'white' }} />
                  ) : finalStep.type === 'sandbox_timeout' ? (
                    <AccessTimeIcon sx={{ fontSize: 14, color: 'white' }} />
                  ) : (
                    <CloseIcon sx={{ fontSize: 14, color: 'white' }} />
                  )}
                </Box>
              </Box>

              {/* Final step label */}
              <Typography
                variant="caption"
                sx={{
                  fontSize: '0.7rem',
                  fontWeight: selectedStepIndex === null ? 700 : 500,
                  color:
                    finalStep.type === 'success'
                      ? (selectedStepIndex === null ? 'text.primary' : 'text.secondary')
                      : finalStep.type === 'stopped' || finalStep.type === 'max_steps_reached'
                        ? 'warning.main'
                        : 'error.main',
                  whiteSpace: 'nowrap',
                }}
              >
                {finalStep.type === 'success' ? 'End' :
                 finalStep.type === 'stopped' ? 'Stopped' :
                 finalStep.type === 'max_steps_reached' ? 'Max Steps' :
                 finalStep.type === 'sandbox_timeout' ? 'Timeout' :
                 'Failed'}
              </Typography>
            </Box>
              )}
            </React.Fragment>
          ))}
        </Box>
      </Box>
    </Box>
  );
};
