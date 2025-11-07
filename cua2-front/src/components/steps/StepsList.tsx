import React, { useRef, useEffect } from 'react';
import { AgentTrace } from '@/types/agent';
import { Box, Typography, Stack, Paper } from '@mui/material';
import { StepCard } from './StepCard';
import { FinalStepCard } from './FinalStepCard';
import { ThinkingStepCard } from './ThinkingStepCard';
import { ConnectionStepCard } from './ConnectionStepCard';
import ListAltIcon from '@mui/icons-material/ListAlt';
import FormatListNumberedIcon from '@mui/icons-material/FormatListNumbered';
import { useAgentStore, selectSelectedStepIndex, selectFinalStep, selectIsConnectingToE2B, selectIsAgentProcessing } from '@/stores/agentStore';

interface StepsListProps {
  trace?: AgentTrace;
}

export const StepsList: React.FC<StepsListProps> = ({ trace }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const selectedStepIndex = useAgentStore(selectSelectedStepIndex);
  const setSelectedStepIndex = useAgentStore((state) => state.setSelectedStepIndex);
  const finalStep = useAgentStore(selectFinalStep);
  const isConnectingToE2B = useAgentStore(selectIsConnectingToE2B);
  const isAgentProcessing = useAgentStore(selectIsAgentProcessing);
  const isScrollingProgrammatically = useRef(false);
  const [showThinkingCard, setShowThinkingCard] = React.useState(false);
  const thinkingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const streamStartTimeRef = useRef<number | null>(null);
  const [showConnectionCard, setShowConnectionCard] = React.useState(false);
  const hasConnectedRef = useRef(false);

  // Check if final step is active (when selectedStepIndex is null and finalStep exists and trace is not running)
  const isFinalStepActive = selectedStepIndex === null && finalStep && !trace?.isRunning;

  // Check if thinking card is active (when in live mode and thinking card is shown)
  const isThinkingCardActive = selectedStepIndex === null && showThinkingCard;

  // Determine the active step index
  // If a specific step is selected, use that
  // If the final step is active, no normal step should be active
  // If the thinking card is active, no normal step should be active
  // Otherwise, show the last step as active
  const activeStepIndex = selectedStepIndex !== null
    ? selectedStepIndex
    : isFinalStepActive
      ? null  // When final step is active, no normal step is active
      : isThinkingCardActive
        ? null  // When thinking card is active, no normal step is active
        : (trace?.steps && trace.steps.length > 0 && trace?.isRunning)
          ? trace.steps.length - 1
          : (trace?.steps && trace.steps.length > 0)
            ? trace.steps.length - 1
            : null;

  // Manage ConnectionStepCard display:
  // - Shows when isConnectingToE2B = true OR when we had a connection
  // - Remains visible even when task is finished (if we have steps or finalStep)
  useEffect(() => {
    if (isConnectingToE2B || isAgentProcessing || (trace?.steps && trace.steps.length > 0) || finalStep) {
      setShowConnectionCard(true);
      hasConnectedRef.current = true;
    }
  }, [isConnectingToE2B, isAgentProcessing, trace?.steps, finalStep]);

  // Manage ThinkingCard display:
  // - Appears 5 seconds AFTER stream starts (isAgentProcessing = true, NOT during isConnectingToE2B)
  // - Remains visible during the entire agent processing
  // - Hides only when agent stops OR a finalStep exists
  useEffect(() => {
    // If stream really starts (isAgentProcessing = true and NOT connecting)
    // And no startTime recorded yet
    if (isAgentProcessing && !isConnectingToE2B && !streamStartTimeRef.current) {
      streamStartTimeRef.current = Date.now();
    }

    // If agent stops OR we have a finalStep, reset and hide
    if (!isAgentProcessing || finalStep) {
      streamStartTimeRef.current = null;
      setShowThinkingCard(false);
      if (thinkingTimeoutRef.current) {
        clearTimeout(thinkingTimeoutRef.current);
        thinkingTimeoutRef.current = null;
      }
      return;
    }

    // If agent is running, not connecting, no finalStep: start 5 second timer
    if (isAgentProcessing && !isConnectingToE2B && !finalStep && streamStartTimeRef.current) {
      // Clean up any existing timeout
      if (thinkingTimeoutRef.current) {
        clearTimeout(thinkingTimeoutRef.current);
      }

      // Calculate elapsed time since stream started
      const elapsedTime = Date.now() - streamStartTimeRef.current;
      const remainingTime = Math.max(0, 5000 - elapsedTime);

      thinkingTimeoutRef.current = setTimeout(() => {
        setShowThinkingCard(true);
      }, remainingTime);
    }

    // Cleanup on unmount or when dependencies change
    return () => {
      if (thinkingTimeoutRef.current) {
        clearTimeout(thinkingTimeoutRef.current);
        thinkingTimeoutRef.current = null;
      }
    };
  }, [isAgentProcessing, isConnectingToE2B, finalStep]);

  // Auto-scroll logic
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    isScrollingProgrammatically.current = true;

    // Use setTimeout to ensure DOM has updated
    setTimeout(() => {
      if (!container) return;

      // LIVE MODE: Always scroll to the bottom (last visible element)
      if (selectedStepIndex === null) {
        // Scroll to bottom
        container.scrollTo({
          top: container.scrollHeight,
          behavior: 'smooth',
        });
      }
      // NON-LIVE MODE: Scroll to selected step
      else {
        const selectedElement = container.querySelector(`[data-step-index="${selectedStepIndex}"]`);
        if (selectedElement) {
          selectedElement.scrollIntoView({
            behavior: 'smooth',
            block: 'center',
          });
        }
      }

      // Reset flag after scroll animation
      setTimeout(() => {
        isScrollingProgrammatically.current = false;
      }, 500);
    }, 100);
  }, [selectedStepIndex, trace?.steps?.length, showThinkingCard, finalStep]);

  // Detect which step is visible when scrolling (steps → timeline)
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !trace?.steps || trace.steps.length === 0) return;

    const handleScroll = () => {
      // Don't update if we're scrolling programmatically
      if (isScrollingProgrammatically.current) return;

      // Don't update if agent is running (stay in live mode)
      if (trace?.isRunning) return;

      const containerRect = container.getBoundingClientRect();
      const containerTop = containerRect.top;
      const containerBottom = containerRect.bottom;
      const containerCenter = containerRect.top + containerRect.height / 2;

      // Check scroll position
      const isAtTop = container.scrollTop <= 5; // 5px tolerance
      const isAtBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 5; // 5px tolerance

      let targetStepIndex: number | null = -1;
      let targetDistance = Infinity;
      let isFinalStepTarget = false;

      if (isAtTop) {
        // At the top: find the highest visible step
        let highestVisibleBottom = Infinity;

        trace.steps.forEach((_, index) => {
          const stepElement = container.querySelector(`[data-step-index="${index}"]`);
          if (stepElement) {
            const stepRect = stepElement.getBoundingClientRect();
            const stepTop = stepRect.top;
            const stepBottom = stepRect.bottom;
            const isVisible = stepTop < containerBottom && stepBottom > containerTop;

            if (isVisible && stepTop < highestVisibleBottom) {
              highestVisibleBottom = stepTop;
              targetStepIndex = index;
              isFinalStepTarget = false;
            }
          }
        });
      } else if (isAtBottom) {
        // At the bottom: find the lowest visible step
        let lowestVisibleTop = -Infinity;

        trace.steps.forEach((_, index) => {
          const stepElement = container.querySelector(`[data-step-index="${index}"]`);
          if (stepElement) {
            const stepRect = stepElement.getBoundingClientRect();
            const stepTop = stepRect.top;
            const stepBottom = stepRect.bottom;
            const isVisible = stepTop < containerBottom && stepBottom > containerTop;

            if (isVisible && stepTop > lowestVisibleTop) {
              lowestVisibleTop = stepTop;
              targetStepIndex = index;
              isFinalStepTarget = false;
            }
          }
        });

        // Check if final step is the lowest visible
        if (finalStep) {
          const finalStepElement = container.querySelector(`[data-step-index="final"]`);
          if (finalStepElement) {
            const finalStepRect = finalStepElement.getBoundingClientRect();
            const finalStepTop = finalStepRect.top;
            const finalStepBottom = finalStepRect.bottom;
            const isVisible = finalStepTop < containerBottom && finalStepBottom > containerTop;

            if (isVisible && finalStepTop > lowestVisibleTop) {
              targetStepIndex = null;
              isFinalStepTarget = true;
            }
          }
        }
      } else {
        // Not at bottom: find the step closest to center
        trace.steps.forEach((_, index) => {
          const stepElement = container.querySelector(`[data-step-index="${index}"]`);
          if (stepElement) {
            const stepRect = stepElement.getBoundingClientRect();
            const stepCenter = stepRect.top + stepRect.height / 2;
            const distance = Math.abs(containerCenter - stepCenter);

            if (distance < targetDistance) {
              targetDistance = distance;
              targetStepIndex = index;
              isFinalStepTarget = false;
            }
          }
        });

        // Check if final step is closest to center
        if (finalStep) {
          const finalStepElement = container.querySelector(`[data-step-index="final"]`);
          if (finalStepElement) {
            const finalStepRect = finalStepElement.getBoundingClientRect();
            const finalStepCenter = finalStepRect.top + finalStepRect.height / 2;
            const distance = Math.abs(containerCenter - finalStepCenter);

            if (distance < targetDistance) {
              targetStepIndex = null;
              isFinalStepTarget = true;
            }
          }
        }
      }

      // Update the selected step if changed
      if (isFinalStepTarget && selectedStepIndex !== null) {
        setSelectedStepIndex(null);
      } else if (!isFinalStepTarget && targetStepIndex !== -1 && targetStepIndex !== selectedStepIndex) {
        setSelectedStepIndex(targetStepIndex);
      }
    };

    // Throttle scroll events
    let scrollTimeout: NodeJS.Timeout;
    const throttledScroll = () => {
      clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(handleScroll, 150);
    };

    container.addEventListener('scroll', throttledScroll);
    return () => {
      container.removeEventListener('scroll', throttledScroll);
      clearTimeout(scrollTimeout);
    };
  }, [trace?.steps, selectedStepIndex, setSelectedStepIndex, finalStep]);

  return (
    <Paper
      elevation={0}
      sx={{
        width: { xs: '100%', md: 320 },
        flexShrink: 0,
        display: 'flex',
        flexDirection: 'column',
        ml: { xs: 0, md: 1.5 },
        mt: { xs: 3, md: 0 },
        overflow: 'hidden',
      }}
    >
      <Box sx={{ px: 2, py: 1.5, borderBottom: '1px solid', borderColor: 'divider', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="h6" sx={{ fontSize: '0.9rem', fontWeight: 700, color: 'text.primary' }}>
          Steps
        </Typography>
        {trace?.traceMetadata && trace.traceMetadata.numberOfSteps > 0 && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0 }}>
            <Typography
              variant="caption"
              sx={{
                fontSize: '0.75rem',
                fontWeight: 700,
                color: 'text.primary',
              }}
            >
              {trace.traceMetadata.numberOfSteps}
            </Typography>
            <Typography
              variant="caption"
              sx={{
                fontSize: '0.75rem',
                fontWeight: 700,
                color: 'text.disabled',
              }}
            >
              /{trace.traceMetadata.maxSteps}
            </Typography>
          </Box>
        )}
      </Box>
      <Box
        ref={containerRef}
        sx={{
          flex: 1,
          overflowY: 'auto',
          minHeight: 0,
          p: 2,
        }}
      >
        {(trace?.steps && trace.steps.length > 0) || finalStep || showThinkingCard || showConnectionCard ? (
          <Stack spacing={2.5}>
            {/* Show connection step card (first item) */}
            {showConnectionCard && (
              <Box data-step-index="connection">
                <ConnectionStepCard isConnecting={isConnectingToE2B} />
              </Box>
            )}

            {/* Show all steps */}
            {trace?.steps && trace.steps.map((step, index) => (
              <Box key={step.stepId} data-step-index={index}>
                <StepCard
                  step={step}
                  index={index}
                  isLatest={index === trace.steps!.length - 1}
                  isActive={index === activeStepIndex}
                />
              </Box>
            ))}

            {/* Show thinking indicator after steps (appears 5 seconds after stream start) */}
            {showThinkingCard && (
              <Box data-step-index="thinking">
                <ThinkingStepCard isActive={isThinkingCardActive} />
              </Box>
            )}

            {/* Show final step card if exists */}
            {finalStep && (
              <Box data-step-index="final">
                <FinalStepCard
                  finalStep={finalStep}
                  isActive={isFinalStepActive}
                />
              </Box>
            )}
          </Stack>
        ) : (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              color: 'text.secondary',
              p: 3,
              textAlign: 'center',
            }}
          >
            <ListAltIcon sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
            <Typography variant="body1" sx={{ fontWeight: 600, mb: 0.5 }}>
              No steps yet
            </Typography>
            <Typography variant="caption" sx={{ fontSize: '0.75rem' }}>
              Steps will appear as the agent progresses
            </Typography>
          </Box>
        )}
      </Box>
    </Paper>
  );
};
