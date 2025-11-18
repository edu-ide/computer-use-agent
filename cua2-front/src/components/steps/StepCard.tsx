import { updateStepEvaluation } from '@/services/api';
import { useAgentStore } from '@/stores/agentStore';
import { AgentStep } from '@/types/agent';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import InputIcon from '@mui/icons-material/Input';
import OutputIcon from '@mui/icons-material/Output';
import ThumbDownIcon from '@mui/icons-material/ThumbDown';
import ThumbUpIcon from '@mui/icons-material/ThumbUp';
import { Accordion, AccordionDetails, AccordionSummary, Box, Card, CardContent, Chip, IconButton, Tooltip, Typography } from '@mui/material';
import React, { useState } from 'react';

interface StepCardProps {
  step: AgentStep;
  index: number;
  isLatest?: boolean;
  isActive?: boolean;
}

export const StepCard: React.FC<StepCardProps> = ({ step, index, isLatest = false, isActive = false }) => {
  const setSelectedStepIndex = useAgentStore((state) => state.setSelectedStepIndex);
  const updateStepEvaluationInStore = useAgentStore((state) => state.updateStepEvaluation);
  const [thoughtExpanded, setThoughtExpanded] = useState(false);
  const [actionsExpanded, setActionsExpanded] = useState(false);
  const [evaluation, setEvaluation] = useState<'like' | 'dislike' | 'neutral'>(step.step_evaluation || 'neutral');
  const [isVoting, setIsVoting] = useState(false);

  const hasMultipleActions = step.actions && step.actions.length > 1;
  const displayedActions = hasMultipleActions && !actionsExpanded
    ? step.actions.slice(0, 1)
    : step.actions;

  const handleClick = () => {
    setSelectedStepIndex(index);
  };

  const handleAccordionClick = (event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent propagation to avoid selecting the step
  };

  const handleVote = async (event: React.MouseEvent, vote: 'like' | 'dislike') => {
    event.stopPropagation(); // Prevent propagation to avoid selecting the step

    if (isVoting) return;

    const newEvaluation = evaluation === vote ? 'neutral' : vote;
    setIsVoting(true);

    try {
      await updateStepEvaluation(step.traceId, step.stepId, newEvaluation);
      setEvaluation(newEvaluation);
      // Update the store so the evaluation is reflected in JSON export
      updateStepEvaluationInStore(step.stepId, newEvaluation);
    } catch (error) {
      console.error('Failed to update step evaluation:', error);
    } finally {
      setIsVoting(false);
    }
  };

  return (
    <Card
      elevation={0}
      onClick={handleClick}
      sx={{
        backgroundColor: 'background.paper',
        border: '1px solid',
        borderColor: (theme) => `${isActive ? theme.palette.primary.main : theme.palette.divider} !important`,
        borderRadius: 1.5,
        transition: 'all 0.2s ease',
        cursor: 'pointer',
        boxShadow: isActive ? (theme) => `0 2px 8px ${theme.palette.mode === 'dark' ? 'rgba(79, 134, 198, 0.3)' : 'rgba(79, 134, 198, 0.2)'}` : 'none',
        '&:hover': {
          borderColor: (theme) => `${theme.palette.primary.main} !important`,
          boxShadow: (theme) => `0 2px 8px ${theme.palette.mode === 'dark' ? 'rgba(79, 134, 198, 0.2)' : 'rgba(79, 134, 198, 0.1)'}`,
        },
      }}
    >
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        {/* Step header */}
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1.5 }}>
          <Typography
            sx={{
              fontSize: '1.5rem',
              fontWeight: 800,
              color: isActive ? 'primary.main' : 'text.primary',
              lineHeight: 1,
            }}
          >
            {index + 1}
          </Typography>
          <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center' }}>
            <Chip
              icon={<AccessTimeIcon sx={{ fontSize: '0.7rem !important' }} />}
              label={`${step.duration.toFixed(1)}s`}
              size="small"
              sx={{
                height: 'auto',
                py: 0.25,
                fontSize: '0.65rem',
                fontWeight: 600,
                backgroundColor: 'action.hover',
                color: 'text.primary',
                '& .MuiChip-icon': { marginLeft: 0.5, color: 'text.secondary' },
              }}
            />
            <Chip
              icon={<InputIcon sx={{ fontSize: '0.7rem !important' }} />}
              label={step.inputTokensUsed.toLocaleString()}
              size="small"
              sx={{
                height: 'auto',
                py: 0.25,
                fontSize: '0.65rem',
                fontWeight: 600,
                backgroundColor: 'action.hover',
                color: 'text.primary',
                '& .MuiChip-icon': { marginLeft: 0.5, color: 'text.secondary' },
              }}
            />
            <Chip
              icon={<OutputIcon sx={{ fontSize: '0.7rem !important' }} />}
              label={step.outputTokensUsed.toLocaleString()}
              size="small"
              sx={{
                height: 'auto',
                py: 0.25,
                fontSize: '0.65rem',
                fontWeight: 600,
                backgroundColor: 'action.hover',
                color: 'text.primary',
                '& .MuiChip-icon': { marginLeft: 0.5, color: 'text.secondary' },
              }}
            />
          </Box>
        </Box>

        {/* Step image */}
        {step.image && (
          <Box
            sx={{
              mb: 1.5,
              borderRadius: 1,
              overflow: 'hidden',
              border: '1px solid',
              borderColor: (theme) => isActive ? theme.palette.primary.main : theme.palette.divider,
              backgroundColor: 'action.hover',
              transition: 'border-color 0.2s ease',
            }}
          >
            <img
              src={step.image}
              alt={`Step ${index + 1}`}
              style={{ width: '100%', height: 'auto', display: 'block' }}
            />
          </Box>
        )}

        {/* Action */}
        {step.actions && step.actions.length > 0 && (
          <Box sx={{ mb: 1.5 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.75, justifyContent: 'space-between' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Typography
                  variant="caption"
                  sx={{
                    fontWeight: 700,
                    color: 'text.secondary',
                    fontSize: '0.65rem',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                  }}
                >
                  Action
                </Typography>
                {hasMultipleActions && (
                  <Tooltip title={actionsExpanded ? 'Show less' : `Show all ${step.actions.length} actions`}>
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        setActionsExpanded(!actionsExpanded);
                      }}
                      sx={{
                        padding: '2px',
                        color: 'text.secondary',
                        '&:hover': {
                          color: 'text.primary',
                          backgroundColor: 'action.hover',
                        },
                      }}
                    >
                      <ExpandMoreIcon
                        sx={{
                          fontSize: 16,
                          transform: actionsExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                          transition: 'transform 0.2s',
                        }}
                      />
                    </IconButton>
                  </Tooltip>
                )}
              </Box>

              {/* Vote buttons */}
              <Box sx={{ display: 'flex', gap: 0.5 }}>
                <Tooltip title={evaluation === 'like' ? 'Remove like' : 'Like this step'}>
                  <IconButton
                    size="small"
                    onClick={(e) => handleVote(e, 'like')}
                    disabled={isVoting}
                    sx={{
                      padding: '2px',
                      color: evaluation === 'like' ? 'success.main' : 'action.disabled',
                      '&:hover': {
                        color: 'success.main',
                        backgroundColor: (theme) => theme.palette.mode === 'dark' ? 'rgba(102, 187, 106, 0.1)' : 'rgba(102, 187, 106, 0.08)',
                      },
                    }}
                  >
                    <ThumbUpIcon sx={{ fontSize: 14 }} />
                  </IconButton>
                </Tooltip>
                <Tooltip title={evaluation === 'dislike' ? 'Remove dislike' : 'Dislike this step'}>
                  <IconButton
                    size="small"
                    onClick={(e) => handleVote(e, 'dislike')}
                    disabled={isVoting}
                    sx={{
                      padding: '2px',
                      color: evaluation === 'dislike' ? 'error.main' : 'action.disabled',
                      '&:hover': {
                        color: 'error.main',
                        backgroundColor: (theme) => theme.palette.mode === 'dark' ? 'rgba(244, 67, 54, 0.1)' : 'rgba(244, 67, 54, 0.08)',
                      },
                    }}
                  >
                    <ThumbDownIcon sx={{ fontSize: 14 }} />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>
            <Box component="ul" sx={{ listStyle: 'none', p: 0, m: 0 }}>
              {displayedActions?.map((action, actionIndex) => (
                <Box
                  key={actionIndex}
                  component="li"
                  sx={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    fontSize: '0.75rem',
                    color: 'text.primary',
                    lineHeight: 1.4,
                    mb: 0.5,
                    '&:last-child': { mb: 0 },
                  }}
                >
                  {/* <Typography
                    component="span"
                    sx={{
                      mr: 0.5,
                      color: 'text.secondary',
                      fontWeight: 700,
                      flexShrink: 0,
                      fontSize: '0.75rem',
                    }}
                  >
                    →
                  </Typography> */}
                  <Typography
                    component="span"
                    sx={{
                      fontSize: '0.75rem',
                      fontWeight: 900,
                      wordBreak: 'break-word',
                    }}
                  >
                    {action.description}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Box>
        )}

        {/* Thought - Accordion */}
        {step.thought && (
          <Accordion
            expanded={thoughtExpanded}
            onChange={(e, expanded) => setThoughtExpanded(expanded)}
            onClick={handleAccordionClick}
            elevation={0}
            disableGutters
            sx={{
              mb: 0.5,
              backgroundColor: 'transparent',
              border: 'none',
              boxShadow: 'none',
              '&:before': { display: 'none' },
              '&.MuiAccordion-root': {
                backgroundColor: 'transparent',
                boxShadow: 'none',
                '&:before': {
                  display: 'none',
                },
              },
              '& .MuiAccordionSummary-root': {
                minHeight: 'auto',
                p: 0,
                backgroundColor: 'transparent',
                '&:hover': {
                  backgroundColor: 'transparent',
                },
                '&.Mui-expanded': {
                  minHeight: 'auto',
                },
              },
              '& .MuiAccordionSummary-content': {
                margin: '0 !important',
              },
              '& .MuiAccordionDetails-root': {
                p: 0,
                pt: 0.5,
                pb: 0,
                backgroundColor: 'transparent',
              },
            }}
          >
            <AccordionSummary
              expandIcon={<ExpandMoreIcon sx={{ fontSize: 16, color: 'text.secondary' }} />}
              sx={{
                flexDirection: 'row',
                border: 'none',
                '& .MuiAccordionSummary-expandIconWrapper': {
                  transform: 'rotate(-90deg)',
                  transition: 'transform 0.2s',
                  '&.Mui-expanded': {
                    transform: 'rotate(0deg)',
                  },
                },
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Typography
                  variant="caption"
                  sx={{
                    fontWeight: 700,
                    color: 'text.secondary',
                    fontSize: '0.65rem',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                  }}
                >
                  Thought
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Typography
                variant="body2"
                sx={{
                  fontSize: '0.75rem',
                  color: 'text.primary',
                  lineHeight: 1.4,
                  pl: 2.5,
                }}
              >
                {step.thought}
              </Typography>
            </AccordionDetails>
          </Accordion>
        )}

        {/* Error */}
        {step.error && (
          <Box sx={{
            mt: 1.5,
            p: 1,
            borderRadius: 1,
            backgroundColor: (theme) => theme.palette.mode === 'dark' ? 'rgba(244, 67, 54, 0.1)' : 'rgba(244, 67, 54, 0.08)',
            border: '1px solid',
            borderColor: 'error.main'
          }}>
            <Typography
              variant="caption"
              sx={{
                fontSize: '0.7rem',
                color: 'error.main',
                fontWeight: 600,
              }}
            >
              Error: {step.error}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};
