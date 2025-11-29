/**
 * VLM 스텝 패널 - 워크플로우 실행 시 실시간 에이전트 스텝 표시
 * - 스텝별 좋아요/싫어요 피드백
 * - Trajectory JSON 다운로드
 */

import React, { useRef, useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Stack,
  Chip,
  keyframes,
  Collapse,
  IconButton,
  Tooltip,
  Button,
  TextField,
  CircularProgress,
  Modal,
  Backdrop,
  Fade,
} from '@mui/material';
import {
  FiMonitor,
  FiMessageSquare,
  FiMousePointer,
  FiEye,
  FiChevronRight,
  FiChevronLeft,
  FiCpu,
  FiDownload,
  FiThumbsUp,
  FiThumbsDown,
  FiCheck,
  FiX,
  FiSave,
  FiMaximize2,
  FiMinimize2,
  FiZoomIn,
  FiZoomOut,
  FiFileText,
} from 'react-icons/fi';
import CloseIcon from '@mui/icons-material/Close';
import { saveTrace } from '@/services/workflowApi';
import { getApiBaseUrl } from '@/config/api';

// 노드 ID -> 한글 이름 매핑
const NODE_DISPLAY_NAMES: Record<string, string> = {
  open_coupang: '쿠팡 열기',
  search_keyword: '키워드 검색',
  analyze_page: '페이지 분석',
  check_next_page: '다음 페이지 확인',
  go_next_page: '다음 페이지 이동',
  complete_collection: '수집 완료',
  error_handler: '오류 처리',
  // YouTube 워크플로우
  open_youtube: '유튜브 열기',
  search_content: '콘텐츠 검색',
  analyze_results: '결과 분석',
  select_video: '영상 선택',
  extract_info: '정보 추출',
};

// 노드별 색상
const NODE_COLORS: Record<string, string> = {
  open_coupang: '#ef4444',    // 빨강
  search_keyword: '#f59e0b',  // 주황
  analyze_page: '#22c55e',    // 초록
  check_next_page: '#3b82f6', // 파랑
  go_next_page: '#8b5cf6',    // 보라
  complete_collection: '#10b981', // 에메랄드
  error_handler: '#dc2626',   // 진한 빨강
  // YouTube 워크플로우
  open_youtube: '#ef4444',
  search_content: '#f59e0b',
  analyze_results: '#22c55e',
  select_video: '#3b82f6',
  extract_info: '#8b5cf6',
};

// 노드별 에이전트 ID (동일 ID면 컨텍스트 공유, 다르면 새 에이전트)
// 현재 구조: 각 노드는 새로운 instruction으로 실행되지만 동일 데스크톱 공유
const NODE_AGENT_INFO: Record<string, { agentId: number; contextShared: boolean; description: string }> = {
  open_coupang: { agentId: 1, contextShared: false, description: '새 대화 시작' },
  search_keyword: { agentId: 1, contextShared: false, description: '새 대화 시작' },
  analyze_page: { agentId: 1, contextShared: false, description: '새 대화 시작' },
  check_next_page: { agentId: 1, contextShared: false, description: '새 대화 시작' },
  go_next_page: { agentId: 1, contextShared: false, description: '새 대화 시작' },
  complete_collection: { agentId: 1, contextShared: false, description: '새 대화 시작' },
  error_handler: { agentId: 1, contextShared: false, description: '새 대화 시작' },
  // YouTube
  open_youtube: { agentId: 1, contextShared: false, description: '새 대화 시작' },
  search_content: { agentId: 1, contextShared: false, description: '새 대화 시작' },
  analyze_results: { agentId: 1, contextShared: false, description: '새 대화 시작' },
  select_video: { agentId: 1, contextShared: false, description: '새 대화 시작' },
  extract_info: { agentId: 1, contextShared: false, description: '새 대화 시작' },
};

const getNodeAgentInfo = (nodeId: string | undefined) => {
  if (!nodeId) return { agentId: 0, contextShared: false, description: '알 수 없음' };
  return NODE_AGENT_INFO[nodeId] || { agentId: 0, contextShared: false, description: '알 수 없음' };
};

const getNodeDisplayName = (nodeId: string | undefined): string => {
  if (!nodeId) return '알 수 없음';
  return NODE_DISPLAY_NAMES[nodeId] || nodeId;
};

const getNodeColor = (nodeId: string | undefined): string => {
  if (!nodeId) return '#64748b';
  return NODE_COLORS[nodeId] || '#64748b';
};

// 펄스 애니메이션
const pulse = keyframes`
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
`;

const spin = keyframes`
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
`;

export interface VLMStep {
  step_number: number;
  timestamp: string;
  screenshot?: string;
  screenshot_after?: string;
  thought?: string;
  action?: string;
  observation?: string;
  error?: string;
  tool_calls?: Array<{ name: string; args: Record<string, unknown> }>;
  node_id?: string;
  evaluation?: 'like' | 'dislike' | 'neutral';
}

interface VLMStepPanelProps {
  steps: VLMStep[];
  currentNode?: string | null;
  isRunning?: boolean;
  collapsed?: boolean;
  onToggle?: () => void;
  executionId?: string | null;
  workflowId?: string;
  onStepEvaluationChange?: (stepIndex: number, evaluation: 'like' | 'dislike' | 'neutral') => void;
  onOverallEvaluationChange?: (evaluation: 'success' | 'failed' | 'not_evaluated') => void;
}

const VLMStepPanel: React.FC<VLMStepPanelProps> = ({
  steps,
  currentNode,
  isRunning = false,
  collapsed = false,
  onToggle,
  executionId,
  workflowId,
  onStepEvaluationChange,
  onOverallEvaluationChange,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [overallEvaluation, setOverallEvaluation] = useState<'success' | 'failed' | 'not_evaluated'>('not_evaluated');
  const [evaluationReason, setEvaluationReason] = useState<string>('');
  const [isSaving, setIsSaving] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [stepEvaluations, setStepEvaluations] = useState<Record<number, 'like' | 'dislike' | 'neutral'>>({});
  const [isExpanded, setIsExpanded] = useState(false);

  // DB에 트레이스 저장
  const handleSaveTrace = async () => {
    if (!executionId || !workflowId || steps.length === 0) return;

    setIsSaving(true);
    try {
      // 첫 번째 스텝에서 에러 찾기
      const errorStep = steps.find((s) => s.error);
      const errorMessage = errorStep?.error || null;
      const errorCause = errorStep ? `Step ${errorStep.step_number}: ${errorStep.action || 'Unknown action'}` : null;

      await saveTrace({
        execution_id: executionId,
        workflow_id: workflowId,
        status: overallEvaluation === 'success' ? 'completed' : overallEvaluation === 'failed' ? 'failed' : 'completed',
        final_state: overallEvaluation === 'success' ? 'success' : overallEvaluation === 'failed' ? 'error' : null,
        error_message: errorMessage,
        error_cause: errorCause,
        user_evaluation: overallEvaluation,
        evaluation_reason: evaluationReason || undefined,
        steps_count: steps.length,
        max_steps: 15,
        steps: steps.map((step, index) => ({
          step_id: `step-${step.step_number}`,
          step_number: step.step_number,
          screenshot: step.screenshot,
          thought: step.thought,
          action: step.action,
          observation: step.observation,
          error: step.error,
          tool_calls: step.tool_calls,
          step_evaluation: stepEvaluations[index] || step.evaluation || 'neutral',
          timestamp: step.timestamp,
        })),
      });
      setIsSaved(true);
    } catch (error) {
      console.error('트레이스 저장 실패:', error);
      alert('트레이스 저장에 실패했습니다.');
    } finally {
      setIsSaving(false);
    }
  };

  // 추론 로그 다운로드 (VLM + Orchestrator 전 과정)
  const handleDownloadReasoningLogs = async () => {
    if (!executionId) return;

    try {
      const apiBase = getApiBaseUrl();
      const response = await fetch(`${apiBase}/workflows/executions/${executionId}/reasoning-logs?format=text`);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: '알 수 없는 오류' }));
        throw new Error(errorData.detail || '추론 로그를 가져올 수 없습니다');
      }
      const text = await response.text();
      const blob = new Blob([text], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `reasoning_logs_${executionId}_${Date.now()}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('추론 로그 다운로드 실패:', error);
      const errorMessage = error instanceof Error ? error.message : '알 수 없는 오류';
      if (errorMessage.includes('찾을 수 없음')) {
        alert('추론 로그를 찾을 수 없습니다.\n실행 중이거나 최근 완료된 워크플로우에서만 다운로드 가능합니다.');
      } else {
        alert(`추론 로그 다운로드 실패: ${errorMessage}`);
      }
    }
  };

  // Trace JSON 다운로드 (기존 trace.json 형식과 호환)
  const handleDownloadTrace = () => {
    // Extract final answer from the last step
    const lastStep = steps[steps.length - 1];
    const finalAnswer = lastStep?.thought || lastStep?.observation || null;

    const traceData = {
      trace: {
        id: executionId,
        timestamp: new Date().toISOString(),
        instruction: workflowId, // 워크플로우 ID를 instruction으로 사용
        modelId: 'vlm-agent',
        isRunning: false,
      },
      completion: {
        status: overallEvaluation === 'success' ? 'success' : overallEvaluation === 'failed' ? 'failure' : 'not_evaluated',
        message: null,
        finalAnswer,
      },
      metadata: {
        traceId: executionId,
        inputTokensUsed: 0,
        outputTokensUsed: 0,
        duration: 0,
        numberOfSteps: steps.length,
        maxSteps: 15,
        completed: true,
        final_state: overallEvaluation === 'success' ? 'success' : overallEvaluation === 'failed' ? 'error' : null,
        user_evaluation: overallEvaluation,
      },
      steps: steps.map((step) => ({
        traceId: executionId,
        stepId: `step-${step.step_number}`,
        error: step.error || null,
        image: step.screenshot || '', // base64 이미지 포함
        thought: step.thought || null,
        actions: step.tool_calls?.map((tc) => ({
          function_name: tc.name,
          parameters: tc.args,
          description: step.action || '',
        })) || null,
        duration: 0,
        inputTokensUsed: 0,
        outputTokensUsed: 0,
        step_evaluation: step.evaluation || 'neutral',
      })),
      exportedAt: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(traceData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trace_${executionId || 'unknown'}_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // 전체 평가 변경
  const handleOverallEvaluation = (evaluation: 'success' | 'failed') => {
    const newEval = overallEvaluation === evaluation ? 'not_evaluated' : evaluation;
    setOverallEvaluation(newEval);
    setIsSaved(false); // 평가 변경 시 저장 상태 리셋
    onOverallEvaluationChange?.(newEval);
  };

  // 스텝 평가 변경
  const handleStepEvaluation = (index: number, evaluation: 'like' | 'dislike' | 'neutral') => {
    setStepEvaluations((prev) => ({ ...prev, [index]: evaluation }));
    setIsSaved(false); // 평가 변경 시 저장 상태 리셋
    onStepEvaluationChange?.(index, evaluation);
  };

  // 새 스텝 추가 시 자동 스크롤
  useEffect(() => {
    if (containerRef.current && steps.length > 0) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [steps.length]);

  if (collapsed) {
    return (
      <Paper
        elevation={2}
        sx={{
          position: 'absolute',
          right: 0,
          top: 56,
          bottom: 0,
          width: 48,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: 'background.paper',
          borderLeft: '1px solid',
          borderColor: 'divider',
          zIndex: 5,
        }}
      >
        <IconButton onClick={onToggle} size="small">
          <FiChevronLeft />
        </IconButton>
        {isRunning && (
          <Box
            sx={{
              mt: 1,
              width: 24,
              height: 24,
              border: '2px solid',
              borderColor: 'primary.main',
              borderTopColor: 'transparent',
              borderRadius: '50%',
              animation: `${spin} 1s linear infinite`,
            }}
          />
        )}
        {steps.length > 0 && (
          <Typography
            variant="caption"
            sx={{
              mt: 1,
              writingMode: 'vertical-rl',
              textOrientation: 'mixed',
              color: 'text.secondary',
            }}
          >
            {steps.length}단계
          </Typography>
        )}
      </Paper>
    );
  }

  return (
    <Paper
      elevation={2}
      sx={{
        position: 'absolute',
        right: 0,
        top: 56,
        bottom: 0,
        width: { xs: '100%', sm: isExpanded ? '80vw' : 360 },
        transition: 'width 0.3s ease',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'background.paper',
        borderLeft: '1px solid',
        borderColor: 'divider',
        zIndex: 5,
      }}
    >
      {/* 헤더 */}
      <Box
        sx={{
          px: 2,
          py: 1.5,
          borderBottom: '1px solid',
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          bgcolor: isRunning ? 'rgba(59, 130, 246, 0.05)' : 'transparent',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <Box
            sx={{
              width: 32,
              height: 32,
              borderRadius: '8px',
              background: isRunning
                ? 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)'
                : '#64748b',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {isRunning ? (
              <Box
                sx={{
                  width: 16,
                  height: 16,
                  border: '2px solid #fff',
                  borderTopColor: 'transparent',
                  borderRadius: '50%',
                  animation: `${spin} 0.8s linear infinite`,
                }}
              />
            ) : (
              <FiCpu color="#fff" size={16} />
            )}
          </Box>
          <Box>
            <Typography variant="subtitle2" fontWeight={700}>
              AI 에이전트
            </Typography>
            {currentNode && (
              <Typography variant="caption" color="text.secondary">
                {currentNode}
              </Typography>
            )}
          </Box>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          {steps.length > 0 && (
            <Chip
              label={`${steps.length}단계`}
              size="small"
              sx={{
                height: 24,
                fontSize: '11px',
                fontWeight: 600,
                bgcolor: isRunning ? 'rgba(59, 130, 246, 0.1)' : 'rgba(100, 116, 139, 0.1)',
                color: isRunning ? '#3b82f6' : '#64748b',
              }}
            />
          )}
          {/* 추론 로그 다운로드 - 실행 중에도 가능 */}
          {steps.length > 0 && executionId && (
            <Tooltip title="추론 로그 다운로드 (전 과정)">
              <IconButton size="small" onClick={handleDownloadReasoningLogs}>
                <FiFileText size={16} />
              </IconButton>
            </Tooltip>
          )}
          {/* 전체 평가 버튼 - 실행 완료 후 */}
          {!isRunning && steps.length > 0 && (
            <>
              <Tooltip title="성공으로 평가">
                <IconButton
                  size="small"
                  onClick={() => handleOverallEvaluation('success')}
                  sx={{
                    color: overallEvaluation === 'success' ? '#22c55e' : '#94a3b8',
                    '&:hover': { color: '#22c55e' },
                  }}
                >
                  <FiCheck size={16} />
                </IconButton>
              </Tooltip>
              <Tooltip title="실패로 평가">
                <IconButton
                  size="small"
                  onClick={() => handleOverallEvaluation('failed')}
                  sx={{
                    color: overallEvaluation === 'failed' ? '#ef4444' : '#94a3b8',
                    '&:hover': { color: '#ef4444' },
                  }}
                >
                  <FiX size={16} />
                </IconButton>
              </Tooltip>
              <Tooltip title="Trace JSON 다운로드">
                <IconButton size="small" onClick={handleDownloadTrace}>
                  <FiDownload size={16} />
                </IconButton>
              </Tooltip>
              <Tooltip title={isSaved ? 'DB에 저장됨' : 'DB에 저장'}>
                <IconButton
                  size="small"
                  onClick={handleSaveTrace}
                  disabled={isSaving || isSaved}
                  sx={{
                    color: isSaved ? '#22c55e' : '#94a3b8',
                    '&:hover': { color: '#3b82f6' },
                  }}
                >
                  {isSaving ? <CircularProgress size={16} /> : <FiSave size={16} />}
                </IconButton>
              </Tooltip>
            </>
          )}
          <Tooltip title={isExpanded ? '축소' : '확대'}>
            <IconButton onClick={() => setIsExpanded(!isExpanded)} size="small">
              {isExpanded ? <FiMinimize2 size={16} /> : <FiMaximize2 size={16} />}
            </IconButton>
          </Tooltip>
          <IconButton onClick={onToggle} size="small">
            <FiChevronRight />
          </IconButton>
        </Box>
      </Box>

      {/* 평가 이유 입력 (실행 완료 후, 평가 선택 시) */}
      {!isRunning && steps.length > 0 && overallEvaluation !== 'not_evaluated' && (
        <Box sx={{ px: 2, py: 1, borderBottom: '1px solid', borderColor: 'divider' }}>
          <TextField
            fullWidth
            size="small"
            placeholder={overallEvaluation === 'success' ? '성공 이유 (선택)' : '실패 원인 (선택)'}
            value={evaluationReason}
            onChange={(e) => setEvaluationReason(e.target.value)}
            sx={{ '& .MuiOutlinedInput-root': { fontSize: '12px' } }}
          />
        </Box>
      )}

      {/* 스텝 목록 - 노드별 그룹화 */}
      <Box
        ref={containerRef}
        sx={{
          flex: 1,
          overflowY: 'auto',
          p: 2,
        }}
      >
        {steps.length > 0 ? (
          <Stack spacing={2}>
            {(() => {
              // 노드별로 스텝 그룹화
              const groupedSteps: { nodeId: string; steps: { step: VLMStep; index: number }[] }[] = [];
              let currentNodeId: string | null = null;

              steps.forEach((step, index) => {
                const nodeId = step.node_id || 'unknown';
                if (nodeId !== currentNodeId) {
                  groupedSteps.push({ nodeId, steps: [] });
                  currentNodeId = nodeId;
                }
                groupedSteps[groupedSteps.length - 1].steps.push({ step, index });
              });

              return groupedSteps.map((group, groupIndex) => {
                const agentInfo = getNodeAgentInfo(group.nodeId);
                return (
                  <Box key={`group-${groupIndex}-${group.nodeId}`}>
                    {/* 노드 헤더 */}
                    <Box
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1,
                        mb: 1,
                        px: 0.5,
                      }}
                    >
                      <Box
                        sx={{
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          bgcolor: getNodeColor(group.nodeId),
                        }}
                      />
                      <Typography
                        variant="caption"
                        fontWeight={700}
                        sx={{ color: getNodeColor(group.nodeId) }}
                      >
                        {getNodeDisplayName(group.nodeId)}
                      </Typography>
                      {/* 에이전트 컨텍스트 정보 */}
                      <Tooltip title={`동일 데스크톱 공유, ${agentInfo.description} (이전 노드 대화 기록 없음)`}>
                        <Chip
                          icon={<FiCpu size={10} />}
                          label={agentInfo.contextShared ? '컨텍스트 공유' : '새 컨텍스트'}
                          size="small"
                          sx={{
                            height: 16,
                            fontSize: '9px',
                            bgcolor: agentInfo.contextShared ? '#dbeafe' : '#fef3c7',
                            color: agentInfo.contextShared ? '#1d4ed8' : '#92400e',
                            fontWeight: 500,
                            '& .MuiChip-icon': {
                              color: agentInfo.contextShared ? '#1d4ed8' : '#92400e',
                              ml: 0.5,
                            },
                          }}
                        />
                      </Tooltip>
                      <Box
                        sx={{
                          flex: 1,
                          height: 1,
                          bgcolor: 'divider',
                          ml: 1,
                        }}
                      />
                      <Chip
                        label={`${group.steps.length}스텝`}
                        size="small"
                        sx={{
                          height: 18,
                          fontSize: '10px',
                          bgcolor: `${getNodeColor(group.nodeId)}15`,
                          color: getNodeColor(group.nodeId),
                          fontWeight: 600,
                        }}
                      />
                    </Box>

                    {/* 해당 노드의 스텝들 */}
                    <Stack spacing={1.5} sx={{ ml: 1.5, borderLeft: '2px solid', borderColor: `${getNodeColor(group.nodeId)}30`, pl: 1.5 }}>
                      {group.steps.map(({ step, index }) => (
                        <StepItem
                          key={`${step.step_number}-${index}`}
                          step={step}
                          isLatest={index === steps.length - 1 && isRunning}
                          index={index}
                          evaluation={stepEvaluations[index] || step.evaluation || 'neutral'}
                          onEvaluationChange={handleStepEvaluation}
                          nodeColor={getNodeColor(group.nodeId)}
                        />
                      ))}
                    </Stack>
                  </Box>
                );
              });
            })()}
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
              textAlign: 'center',
            }}
          >
            <FiMonitor size={40} style={{ opacity: 0.3, marginBottom: 12 }} />
            <Typography variant="body2" fontWeight={500}>
              {isRunning ? '에이전트 준비 중...' : '스텝 없음'}
            </Typography>
            <Typography variant="caption" color="text.disabled" sx={{ mt: 0.5 }}>
              {isRunning
                ? '곧 에이전트 동작이 시작됩니다'
                : '워크플로우 실행 시 스텝이 표시됩니다'}
            </Typography>
          </Box>
        )}
      </Box>
    </Paper>
  );
};

// 개별 스텝 아이템
interface StepItemProps {
  step: VLMStep;
  isLatest: boolean;
  index: number;
  evaluation: 'like' | 'dislike' | 'neutral';
  onEvaluationChange?: (index: number, evaluation: 'like' | 'dislike' | 'neutral') => void;
  nodeColor?: string;
}

const StepItem: React.FC<StepItemProps> = ({ step, isLatest, index, evaluation, onEvaluationChange, nodeColor = '#64748b' }) => {
  const [expanded, setExpanded] = useState(true);
  const [screenshotModalOpen, setScreenshotModalOpen] = useState(false);
  const [selectedScreenshot, setSelectedScreenshot] = useState<string | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);

  const handleVote = (e: React.MouseEvent, vote: 'like' | 'dislike') => {
    e.stopPropagation();
    const newEval = evaluation === vote ? 'neutral' : vote;
    onEvaluationChange?.(index, newEval);
  };

  const handleScreenshotClick = (e: React.MouseEvent, src: string) => {
    e.stopPropagation();
    setZoomLevel(1);
    setSelectedScreenshot(src);
    setScreenshotModalOpen(true);
  };

  const handleZoomIn = () => {
    setZoomLevel((prev) => Math.min(prev + 0.25, 3));
  };

  const handleZoomOut = () => {
    setZoomLevel((prev) => Math.max(prev - 0.25, 0.5));
  };

  const getScreenshotSrc = (base64Str?: string) => {
    if (!base64Str) return '';
    return base64Str.startsWith('data:') ? base64Str : `data:image/png;base64,${base64Str}`;
  };

  return (
    <Paper
      variant="outlined"
      sx={{
        overflow: 'hidden',
        borderColor: isLatest ? nodeColor : 'divider',
        borderWidth: isLatest ? 2 : 1,
        animation: isLatest ? `${pulse} 2s ease infinite` : 'none',
      }}
    >
      {/* 스텝 헤더 */}
      <Box
        onClick={() => setExpanded(!expanded)}
        sx={{
          px: 1.5,
          py: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          cursor: 'pointer',
          bgcolor: isLatest ? `${nodeColor}10` : 'rgba(0, 0, 0, 0.02)',
          '&:hover': { bgcolor: 'action.hover' },
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 24,
              height: 24,
              borderRadius: '50%',
              bgcolor: isLatest ? nodeColor : `${nodeColor}99`,
              color: '#fff',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 11,
              fontWeight: 700,
            }}
          >
            {step.step_number}
          </Box>
          <Typography variant="body2" fontWeight={600} sx={{ color: isLatest ? nodeColor : 'text.primary' }}>
            Step {step.step_number}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          {/* 스텝별 좋아요/싫어요 */}
          <Tooltip title="좋아요">
            <IconButton
              size="small"
              onClick={(e) => handleVote(e, 'like')}
              sx={{
                p: 0.5,
                color: evaluation === 'like' ? '#22c55e' : '#94a3b8',
                '&:hover': { color: '#22c55e' },
              }}
            >
              <FiThumbsUp size={12} />
            </IconButton>
          </Tooltip>
          <Tooltip title="싫어요">
            <IconButton
              size="small"
              onClick={(e) => handleVote(e, 'dislike')}
              sx={{
                p: 0.5,
                color: evaluation === 'dislike' ? '#ef4444' : '#94a3b8',
                '&:hover': { color: '#ef4444' },
              }}
            >
              <FiThumbsDown size={12} />
            </IconButton>
          </Tooltip>
          <Typography variant="caption" color="text.secondary" sx={{ ml: 0.5 }}>
            {new Date(step.timestamp).toLocaleTimeString()}
          </Typography>
        </Box>
      </Box>

      {/* 스텝 내용 */}
      <Collapse in={expanded}>
        <Box sx={{ p: 1.5 }}>
          {/* 스크린샷 (Before / After) */}
          {(step.screenshot || step.screenshot_after) && (
            <Box sx={{ mb: 1.5 }}>
              {step.screenshot && step.screenshot_after ? (
                <Box sx={{ display: 'flex', gap: 1 }}>
                  {/* Before */}
                  <Box sx={{ flex: 1, position: 'relative' }}>
                    <Typography variant="caption" sx={{ display: 'block', mb: 0.5, color: 'text.secondary', fontWeight: 600 }}>
                      Action 전
                    </Typography>
                    <Box
                      component="img"
                      src={getScreenshotSrc(step.screenshot)}
                      alt="Action 전"
                      onClick={(e) => handleScreenshotClick(e, getScreenshotSrc(step.screenshot))}
                      sx={{
                        width: '100%',
                        borderRadius: 1,
                        border: '1px solid',
                        borderColor: 'divider',
                        cursor: 'pointer',
                        '&:hover': { opacity: 0.9 },
                      }}
                    />
                  </Box>
                  {/* After */}
                  <Box sx={{ flex: 1, position: 'relative' }}>
                    <Typography variant="caption" sx={{ display: 'block', mb: 0.5, color: 'text.secondary', fontWeight: 600 }}>
                      Action 후
                    </Typography>
                    <Box
                      component="img"
                      src={getScreenshotSrc(step.screenshot_after)}
                      alt="Action 후"
                      onClick={(e) => handleScreenshotClick(e, getScreenshotSrc(step.screenshot_after))}
                      sx={{
                        width: '100%',
                        borderRadius: 1,
                        border: '1px solid',
                        borderColor: 'divider',
                        cursor: 'pointer',
                        '&:hover': { opacity: 0.9 },
                      }}
                    />
                  </Box>
                </Box>
              ) : (
                // Single image (usually Before)
                <Box sx={{ position: 'relative' }}>
                  <Box
                    component="img"
                    src={getScreenshotSrc(step.screenshot || step.screenshot_after)}
                    alt={`${step.step_number}단계 스크린샷`}
                    onClick={(e) => handleScreenshotClick(e, getScreenshotSrc(step.screenshot || step.screenshot_after))}
                    sx={{
                      width: '100%',
                      borderRadius: 1,
                      border: '1px solid',
                      borderColor: 'divider',
                      cursor: 'pointer',
                      transition: 'transform 0.2s, box-shadow 0.2s',
                      '&:hover': {
                        transform: 'scale(1.02)',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                      },
                    }}
                  />
                  <Tooltip title="확대 보기">
                    <IconButton
                      size="small"
                      onClick={(e) => handleScreenshotClick(e, getScreenshotSrc(step.screenshot || step.screenshot_after))}
                      sx={{
                        position: 'absolute',
                        top: 4,
                        right: 4,
                        bgcolor: 'rgba(0,0,0,0.5)',
                        color: 'white',
                        '&:hover': { bgcolor: 'rgba(0,0,0,0.7)' },
                      }}
                    >
                      <FiMaximize2 size={14} />
                    </IconButton>
                  </Tooltip>
                </Box>
              )}
            </Box>
          )}

          {/* 스크린샷 확대 모달 */}
          <Modal
            open={screenshotModalOpen}
            onClose={() => setScreenshotModalOpen(false)}
            closeAfterTransition
            slots={{ backdrop: Backdrop }}
            slotProps={{ backdrop: { timeout: 300, sx: { bgcolor: 'rgba(0,0,0,0.85)' } } }}
          >
            <Fade in={screenshotModalOpen}>
              <Box
                sx={{
                  position: 'fixed',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  outline: 'none',
                }}
              >
                {/* 모달 헤더 */}
                <Box
                  sx={{
                    px: 3,
                    py: 1.5,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    bgcolor: 'rgba(0,0,0,0.9)',
                    borderBottom: '1px solid rgba(255,255,255,0.1)',
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Box
                      sx={{
                        width: 32,
                        height: 32,
                        borderRadius: '50%',
                        bgcolor: nodeColor,
                        color: '#fff',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 14,
                        fontWeight: 700,
                      }}
                    >
                      {step.step_number}
                    </Box>
                    <Box>
                      <Typography variant="subtitle1" fontWeight={600} sx={{ color: '#fff' }}>
                        Step {step.step_number}
                      </Typography>
                      {step.node_id && (
                        <Typography variant="caption" sx={{ color: nodeColor }}>
                          {getNodeDisplayName(step.node_id)}
                        </Typography>
                      )}
                    </Box>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Tooltip title="축소 (-)">
                      <IconButton
                        size="small"
                        onClick={handleZoomOut}
                        disabled={zoomLevel <= 0.5}
                        sx={{ color: '#fff', '&:disabled': { color: 'rgba(255,255,255,0.3)' } }}
                      >
                        <FiZoomOut size={20} />
                      </IconButton>
                    </Tooltip>
                    <Box
                      sx={{
                        minWidth: 60,
                        textAlign: 'center',
                        bgcolor: 'rgba(255,255,255,0.1)',
                        borderRadius: 1,
                        px: 1.5,
                        py: 0.5,
                      }}
                    >
                      <Typography variant="body2" sx={{ color: '#fff', fontWeight: 600 }}>
                        {Math.round(zoomLevel * 100)}%
                      </Typography>
                    </Box>
                    <Tooltip title="확대 (+)">
                      <IconButton
                        size="small"
                        onClick={handleZoomIn}
                        disabled={zoomLevel >= 3}
                        sx={{ color: '#fff', '&:disabled': { color: 'rgba(255,255,255,0.3)' } }}
                      >
                        <FiZoomIn size={20} />
                      </IconButton>
                    </Tooltip>
                    <Box sx={{ width: 1, height: 24, bgcolor: 'rgba(255,255,255,0.2)', mx: 1 }} />
                    <Tooltip title="닫기 (ESC)">
                      <IconButton
                        size="small"
                        onClick={() => setScreenshotModalOpen(false)}
                        sx={{ color: '#fff', '&:hover': { bgcolor: 'rgba(255,255,255,0.1)' } }}
                      >
                        <CloseIcon />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>

                {/* 이미지 컨테이너 - 전체 화면 */}
                <Box
                  sx={{
                    flex: 1,
                    overflow: 'auto',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    p: 3,
                    bgcolor: '#0a0a0a',
                  }}
                >
                  {selectedScreenshot && (
                    <Box
                      component="img"
                      src={selectedScreenshot}
                      alt={`${step.step_number}단계 스크린샷`}
                      sx={{
                        maxWidth: zoomLevel === 1 ? '100%' : 'none',
                        maxHeight: zoomLevel === 1 ? '100%' : 'none',
                        width: zoomLevel !== 1 ? `${zoomLevel * 100}%` : 'auto',
                        objectFit: 'contain',
                        borderRadius: 1,
                        boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
                        transition: 'width 0.2s ease-out',
                      }}
                    />
                  )}
                </Box>

                {/* 하단 정보 바 */}
                {(step.action || step.thought) && (
                  <Box
                    sx={{
                      px: 3,
                      py: 1.5,
                      bgcolor: 'rgba(0,0,0,0.9)',
                      borderTop: '1px solid rgba(255,255,255,0.1)',
                      maxHeight: 100,
                      overflow: 'auto',
                    }}
                  >
                    {step.action && (
                      <Typography variant="body2" sx={{ color: '#3b82f6', fontFamily: 'monospace' }}>
                        <strong>Action:</strong> {step.action}
                      </Typography>
                    )}
                    {step.thought && (
                      <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.7)', display: 'block', mt: 0.5 }}>
                        {step.thought.length > 150 ? `${step.thought.slice(0, 150)}...` : step.thought}
                      </Typography>
                    )}
                  </Box>
                )}
              </Box>
            </Fade>
          </Modal>

          {/* 생각 */}
          {step.thought && (
            <Box sx={{ mb: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                <FiMessageSquare size={12} color="#8b5cf6" />
                <Typography variant="caption" color="#8b5cf6" fontWeight={600}>
                  생각
                </Typography>
              </Box>
              <Typography
                variant="body2"
                sx={{
                  fontSize: '12px',
                  color: 'text.secondary',
                  bgcolor: 'rgba(139, 92, 246, 0.05)',
                  p: 1,
                  borderRadius: 1,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  lineHeight: 1.5,
                  maxHeight: 150,
                  overflow: 'auto',
                }}
              >
                {step.thought}
              </Typography>
            </Box>
          )}

          {/* 액션 */}
          {step.action && (
            <Box sx={{ mb: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                <FiMousePointer size={12} color="#3b82f6" />
                <Typography variant="caption" color="#3b82f6" fontWeight={600}>
                  액션
                </Typography>
              </Box>
              <Typography
                variant="body2"
                sx={{
                  fontSize: '12px',
                  color: 'text.primary',
                  bgcolor: 'rgba(59, 130, 246, 0.05)',
                  p: 1,
                  borderRadius: 1,
                  fontFamily: 'monospace',
                }}
              >
                {step.action}
              </Typography>
            </Box>
          )}

          {/* 관찰 */}
          {step.observation && (
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                <FiEye size={12} color="#22c55e" />
                <Typography variant="caption" color="#22c55e" fontWeight={600}>
                  관찰
                </Typography>
              </Box>
              <Typography
                variant="body2"
                sx={{
                  fontSize: '12px',
                  color: 'text.secondary',
                  bgcolor: 'rgba(34, 197, 94, 0.05)',
                  p: 1,
                  borderRadius: 1,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  maxHeight: 100,
                  overflow: 'auto',
                }}
              >
                {step.observation}
              </Typography>
            </Box>
          )}

          {/* 에러 */}
          {step.error && (
            <Box sx={{ mt: 1 }}>
              <Typography
                variant="body2"
                sx={{
                  fontSize: '12px',
                  color: '#ef4444',
                  bgcolor: 'rgba(239, 68, 68, 0.05)',
                  p: 1,
                  borderRadius: 1,
                }}
              >
                {step.error}
              </Typography>
            </Box>
          )}
        </Box>
      </Collapse>
    </Paper>
  );
};

export default VLMStepPanel;
