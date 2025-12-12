/**
 * 워크플로우 상세/실행 페이지 - XState 상태 머신 기반
 */

import React, { useEffect, useCallback, useRef } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useMachine } from '@xstate/react';
import {
  Box,
  Typography,
  Button,
  IconButton,
  Chip,
  Divider,
  Alert,
  CircularProgress,
  Tooltip,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import YouTubeIcon from '@mui/icons-material/YouTube';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import InventoryIcon from '@mui/icons-material/Inventory';
import DarkModeOutlined from '@mui/icons-material/DarkModeOutlined';
import LightModeOutlined from '@mui/icons-material/LightModeOutlined';
import WifiIcon from '@mui/icons-material/Wifi';
import WifiOffIcon from '@mui/icons-material/WifiOff';
import { FiPlay, FiPause, FiSquare, FiSkipForward, FiRefreshCw, FiX, FiCpu, FiSettings } from 'react-icons/fi';
import ReactJson from 'react-json-view';

import { selectIsDarkMode, useAgentStore } from '@/stores/agentStore';
import {
  WorkflowGraph,
  NodeDetailModal,
  ParameterModal,
  ParameterConfig,
  VLMStepPanel,
  AgentActivityPanel,
  ProductListModal,
} from '@/components/workflow';
import { getWorkflowDetail } from '@/services/workflowApi';
import {
  workflowMachine,
  isRunningState,
  WorkflowState,
} from '@/machines/workflowMachine';
import VisibilityIcon from '@mui/icons-material/Visibility';
import CenterFocusStrongIcon from '@mui/icons-material/CenterFocusStrong';

// The following imports and types are based on the provided "Code Edit" and assume a refactor to a new state management and component structure.
// If these are not available in the project, this change will introduce errors.
import { WorkflowNodeData } from '@/components/workflow/WorkflowNode'; // Assuming this path based on context

// 아이콘 매핑
const ICON_MAP: Record<string, React.ElementType> = {
  ShoppingCart: ShoppingCartIcon,
  YouTube: YouTubeIcon,
  AccountTree: AccountTreeIcon,
};

/**
 * WebSocket URL 생성
 */
const getWorkflowWsUrl = (executionId: string): string => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.hostname;
  const port = import.meta.env.VITE_API_PORT || '8000';

  if (import.meta.env.PROD) {
    return `${protocol}//${window.location.host}/ws/workflow/${executionId}`;
  }

  return `${protocol}//${host}:${port}/ws/workflow/${executionId}`;
};

const WorkflowDetail: React.FC = () => {
  const navigate = useNavigate();
  const { workflowId } = useParams<{ workflowId: string }>();
  const isDarkMode = useAgentStore(selectIsDarkMode);
  const toggleDarkMode = useAgentStore((state) => state.toggleDarkMode);

  // XState 머신 사용
  const [state, send] = useMachine(workflowMachine);
  const {
    workflow,
    parameters,
    executionId,
    currentNode,
    completedNodes,
    failedNodes,
    vlmSteps,
    lastScreenshot,
    error,
    wsConnected,
    // 현재 노드 상세 정보
    currentNodeStartTime,
    currentThought,
    currentAction,
    currentObservation,
    stepCount,
    // 에러 추적 정보
    consecutiveErrors,
    lastError,
  } = state.context;

  // 현재 상태 값
  const machineState = state.value as WorkflowState;
  const isRunning = isRunningState(machineState);

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null);

  // UI 상태
  const [parameterModalOpen, setParameterModalOpen] = React.useState(false);
  const [selectedNodeId, setSelectedNodeId] = React.useState<string | null>(null);
  const [nodeModalOpen, setNodeModalOpen] = React.useState(false);
  const [showLiveScreen, setShowLiveScreen] = React.useState(false);
  const [focusMode, setFocusMode] = React.useState(true);
  const [stepPanelCollapsed, setStepPanelCollapsed] = React.useState(false);
  const [activityPanelCollapsed, setActivityPanelCollapsed] = React.useState(false);
  const [loading, setLoading] = React.useState(true);
  const [productModalOpen, setProductModalOpen] = React.useState(false);

  // 워크플로우 상세 로드
  useEffect(() => {
    if (workflowId) {
      loadWorkflowDetail(workflowId);
    }
  }, [workflowId]);

  const loadWorkflowDetail = async (id: string) => {
    setLoading(true);
    try {
      const result = await getWorkflowDetail(id);
      send({ type: 'SET_WORKFLOW', workflow: result });

      // 기본 파라미터 설정
      const defaultParams: Record<string, unknown> = {};
      result.config.parameters.forEach((param) => {
        if (param.default !== undefined) {
          defaultParams[param.name] = param.default;
        }
      });

      // localStorage에서 저장된 파라미터 읽기
      const storageKey = `workflow_params_${id}`;
      let savedParams: Record<string, unknown> = {};
      try {
        const savedStr = localStorage.getItem(storageKey);
        if (savedStr) {
          savedParams = JSON.parse(savedStr);
        }
      } catch (e) {
        console.warn('Failed to load saved params:', e);
      }

      // 기본값과 저장된 값 병합 (저장된 값 우선)
      const mergedParams = { ...defaultParams, ...savedParams };
      send({ type: 'SET_PARAMETERS', parameters: mergedParams });
    } catch (err) {
      console.error('워크플로우 로드 실패:', err);
    } finally {
      setLoading(false);
    }
  };

  // WebSocket 연결 관리
  useEffect(() => {
    // 실행 중이고 executionId가 있을 때만 WebSocket 연결
    if (!isRunning || !executionId) {
      if (wsRef.current) {
        wsRef.current.close(1000, 'Not running');
        wsRef.current = null;
      }
      return;
    }

    const url = getWorkflowWsUrl(executionId);
    console.log(`WebSocket 연결: ${url}`);

    const ws = new WebSocket(url);

    ws.onopen = () => {
      console.log('WebSocket 연결됨');
      send({ type: 'WS_CONNECTED' });
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // 디버깅: WebSocket 메시지 로깅
        console.log('[WebSocket] Message received:', {
          type: data.type,
          status: data.status,
          current_node: data.current_node,
          completed_nodes: data.completed_nodes,
        });

        switch (data.type) {
          case 'status':
            console.log('[WebSocket] Sending WS_STATUS event to machine');
            send({
              type: 'WS_STATUS',
              status: {
                execution_id: data.execution_id,
                workflow_id: data.workflow_id,
                status: data.status,
                current_node: data.current_node,
                completed_nodes: data.completed_nodes,
                failed_nodes: data.failed_nodes,
                data: data.data,
                error: data.error,
                start_time: data.start_time,
                end_time: data.end_time,
              },
              allSteps: data.all_steps,
              lastScreenshot: data.last_screenshot,
              // 현재 노드 상세 정보
              currentNodeStartTime: data.current_node_start_time,
              currentThought: data.current_thought,
              currentAction: data.current_action,
              currentObservation: data.current_observation,
              stepCount: data.step_count,
              // 에러 추적 정보
              consecutiveErrors: data.consecutive_errors,
              lastError: data.last_error,
            });
            break;
          case 'step':
            send({ type: 'WS_STEP', step: data.step });
            break;
          case 'complete':
            send({ type: 'WS_COMPLETE', status: data.status, error: data.error });
            break;
          case 'error':
            send({ type: 'WS_ERROR', message: data.message });
            break;
        }
      } catch (err) {
        console.error('WebSocket 메시지 파싱 오류:', err);
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket 오류:', err);
    };

    ws.onclose = (event) => {
      console.log('WebSocket 연결 해제:', event.code);
      send({ type: 'WS_DISCONNECTED' });
    };

    wsRef.current = ws;

    return () => {
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmount');
        wsRef.current = null;
      }
    };
  }, [isRunning, executionId, send]);

  // 노드 클릭 핸들러
  const handleNodeClick = useCallback(
    (nodeId: string) => {
      const node = workflow?.nodes.find((n) => n.id === nodeId);
      if (!node) return;

      // 클릭 가능 조건:
      // 1. node.clickable이 true인 경우
      // 2. VLM 타입 노드
      // 3. 실행 완료/실패/중지 후 모든 노드
      // 4. 현재 완료된 노드 또는 실패한 노드
      const isVLMNode = node.type === 'vlm';
      const isClickableNode = node.clickable === true;
      const isExecutionFinished = ['completed', 'failed', 'stopped'].includes(machineState);
      const isCompletedOrFailed = completedNodes.includes(nodeId) || failedNodes.includes(nodeId);

      if (isClickableNode || isVLMNode || isExecutionFinished || isCompletedOrFailed) {
        setSelectedNodeId(nodeId);
        setNodeModalOpen(true);
      }
    },
    [workflow, machineState, completedNodes, failedNodes]
  );

  // 시작 핸들러
  const handleStart = useCallback(() => {
    if (!workflow) return;
    send({
      type: 'START',
      workflowId: workflow.config.id,
      parameters,
    });
    setStepPanelCollapsed(false);
  }, [workflow, parameters, send]);

  // 중지 핸들러
  const handleStop = useCallback(() => {
    send({ type: 'STOP' });
  }, [send]);

  // 파라미터 변경 핸들러
  const handleParameterChange = useCallback(
    (name: string, value: unknown) => {
      const updatedParams = { ...parameters, [name]: value };
      send({
        type: 'SET_PARAMETERS',
        parameters: updatedParams,
      });

      // localStorage에 저장
      if (workflowId) {
        try {
          const storageKey = `workflow_params_${workflowId}`;
          localStorage.setItem(storageKey, JSON.stringify(updatedParams));
        } catch (e) {
          console.warn('Failed to save params:', e);
        }
      }
    },
    [parameters, send, workflowId]
  );

  // 상태 표시
  const getStatusColor = () => {
    switch (machineState) {
      case 'running':
      case 'connecting':
        return 'primary';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'stopped':
      case 'stopping':
        return 'warning';
      default:
        return 'default';
    }
  };

  const getStatusLabel = () => {
    switch (machineState) {
      case 'idle':
        return '대기 중';
      case 'starting':
        return '시작 중...';
      case 'connecting':
        return '연결 중...';
      case 'running':
        return '실행 중';
      case 'stopping':
        return '중지 중...';
      case 'completed':
        return '완료';
      case 'failed':
        return '실패';
      case 'stopped':
        return '중지됨';
      default:
        return machineState;
    }
  };

  const getWorkflowDefinition = () => {
    if (!workflow) return null;

    return {
      nodes: workflow.nodes.map((node) => ({
        id: node.id,
        name: node.name,
        description: node.description,
        status: node.status,
        nodeType: node.type,
        // 재사용/메모리 설정
        reusable: node.reusable,
        reuse_trace: node.reuse_trace,
        share_memory: node.share_memory,
        cache_key_params: node.cache_key_params,
        // 에이전트 정보
        agent_type: node.agent_type,
        model_id: node.model_id,
        // 클릭 가능 여부 - VLM 노드이거나 명시적으로 clickable인 경우
        clickable: node.clickable ?? (node.type === 'vlm'),
        // 시간 설정
        timeout_sec: node.timeout_sec,
        avg_duration_sec: node.avg_duration_sec,
      })),
      edges: workflow.edges.map((edge) => ({
        source: edge.source,
        target: edge.target,
        type: edge.type,
      })),
      startNode: workflow.start_node,
    };
  };

  const getExecutionState = () => {
    // idle 상태에서만 null 반환
    if (machineState === 'idle') return null;

    // 실행 중이거나 완료 상태에서는 항상 상태 반환
    return {
      currentNode: currentNode || null,
      completedNodes: completedNodes || [],
      failedNodes: failedNodes || [],
      // 현재 노드 상세 정보
      currentNodeStartTime: currentNodeStartTime || undefined,
      currentThought: currentThought || undefined,
      currentAction: currentAction || undefined,
      currentObservation: currentObservation || undefined,
      stepCount: stepCount || undefined,
      // 에러 추적 정보
      consecutiveErrors: consecutiveErrors || undefined,
      lastError: lastError || undefined,
    };
  };

  // 디버깅: 상태 변화 로깅
  useEffect(() => {
    console.log('[WorkflowDetail] State changed:', {
      machineState,
      isRunning,
      currentNode,
      completedNodes,
      failedNodes,
      wsConnected,
      executionId,
    });
  }, [machineState, isRunning, currentNode, completedNodes, failedNodes, wsConnected, executionId]);

  const getParameterConfig = (): ParameterConfig[] => {
    if (!workflow) return [];
    return workflow.config.parameters.map((param) => ({
      name: param.name,
      label: param.label,
      value: parameters[param.name],
    }));
  };

  if (loading) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '100vh',
          bgcolor: 'background.default',
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  if (!workflow) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '100vh',
          bgcolor: 'background.default',
          gap: 2,
        }}
      >
        <Typography variant="h6" color="text.secondary">
          워크플로우를 찾을 수 없습니다
        </Typography>
        <Button variant="contained" onClick={() => navigate('/workflows')}>
          목록으로 돌아가기
        </Button>
      </Box>
    );
  }

  const IconComponent = ICON_MAP[workflow.config.icon] || AccountTreeIcon;

  return (
    <Box
      sx={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'background.default',
      }}
    >
      {/* 상단 툴바 */}
      <Box
        sx={{
          height: 56,
          px: 2,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid',
          borderColor: 'divider',
          bgcolor: 'background.paper',
          flexShrink: 0,
        }}
      >
        {/* 왼쪽: 뒤로가기 + 워크플로우 이름 */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <IconButton onClick={() => navigate('/workflows')} size="small">
            <ArrowBackIcon />
          </IconButton>

          <Divider orientation="vertical" flexItem />
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box
              sx={{
                width: 28,
                height: 28,
                borderRadius: 1,
                bgcolor: workflow.config.color,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <IconComponent sx={{ color: 'white', fontSize: 16 }} />
            </Box>
            <Typography variant="subtitle1" fontWeight={600}>
              {workflow.config.name}
            </Typography>
          </Box>

          {/* 실행/중지 버튼 */}
          <Divider orientation="vertical" flexItem />
          {!isRunning ? (
            <Button
              variant="contained"
              size="small"
              startIcon={<PlayArrowIcon />}
              onClick={handleStart}
              sx={{
                textTransform: 'none',
                fontWeight: 600,
                background: 'linear-gradient(135deg, #1677ff 0%, #0958d9 100%)',
              }}
            >
              실행
            </Button>
          ) : (
            <Button
              variant="outlined"
              size="small"
              color="error"
              startIcon={<StopIcon />}
              onClick={handleStop}
              disabled={machineState === 'stopping'}
              sx={{ textTransform: 'none', fontWeight: 600 }}
            >
              {machineState === 'stopping' ? '중지 중...' : '중지'}
            </Button>
          )}
        </Box>

        {/* 중앙: 실행 상태 */}
        {machineState !== 'idle' && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Chip
              label={getStatusLabel()}
              color={getStatusColor()}
              size="small"
              icon={
                isRunning ? (
                  <CircularProgress size={14} color="inherit" />
                ) : undefined
              }
            />
            {currentNode && (
              <Typography variant="body2" color="text.secondary">
                현재: <strong>{currentNode}</strong>
              </Typography>
            )}
          </Box>
        )}

        {/* 오른쪽: 액션 버튼 */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {isRunning && (
            <>
              {/* WebSocket 연결 상태 */}
              <Tooltip
                title={wsConnected ? 'WebSocket 연결됨' : 'WebSocket 연결 중...'}
              >
                <IconButton
                  size="small"
                  sx={{ color: wsConnected ? '#22c55e' : '#94a3b8' }}
                >
                  {wsConnected ? (
                    <WifiIcon fontSize="small" />
                  ) : (
                    <WifiOffIcon fontSize="small" />
                  )}
                </IconButton>
              </Tooltip>
              <Tooltip
                title={focusMode ? '포커스 모드 끄기' : '포커스 모드 켜기'}
              >
                <Button
                  onClick={() => setFocusMode(!focusMode)}
                  size="small"
                  variant={focusMode ? 'contained' : 'outlined'}
                  color={focusMode ? 'primary' : 'inherit'}
                  startIcon={<CenterFocusStrongIcon />}
                  sx={{ textTransform: 'none' }}
                >
                  포커스
                </Button>
              </Tooltip>
              <Tooltip
                title={showLiveScreen ? '그래프 보기' : '실시간 화면 보기'}
              >
                <Button
                  onClick={() => setShowLiveScreen(!showLiveScreen)}
                  size="small"
                  variant={showLiveScreen ? 'contained' : 'outlined'}
                  startIcon={<VisibilityIcon />}
                  sx={{ textTransform: 'none' }}
                >
                  {showLiveScreen ? '그래프' : '화면'}
                </Button>
              </Tooltip>
            </>
          )}
          <Button
            onClick={() => setProductModalOpen(true)}
            size="small"
            startIcon={<InventoryIcon />}
            sx={{ textTransform: 'none' }}
          >
            상품 목록
          </Button>
          <IconButton onClick={toggleDarkMode} size="small">
            {isDarkMode ? <LightModeOutlined /> : <DarkModeOutlined />}
          </IconButton>
        </Box>
      </Box>

      {/* 메인 영역 - 그래프 전체 화면 */}
      <Box
        sx={{
          flex: 1,
          position: 'relative',
          bgcolor: '#f8fafc',
          overflow: 'hidden',
        }}
      >
        {/* 에러 알림 */}
        {error && (
          <Alert
            severity="error"
            onClose={() => send({ type: 'RESET' })}
            sx={{
              position: 'absolute',
              top: 16,
              left: '50%',
              transform: 'translateX(-50%)',
              zIndex: 10,
              maxWidth: 500,
            }}
          >
            {error}
          </Alert>
        )}

        {/* 완료 알림 */}
        {machineState === 'completed' && (
          <Alert
            severity="success"
            sx={{
              position: 'absolute',
              top: 16,
              left: '50%',
              transform: 'translateX(-50%)',
              zIndex: 10,
            }}
          >
            워크플로우가 성공적으로 완료되었습니다.
          </Alert>
        )}

        {/* 실시간 화면 보기 */}
        {showLiveScreen && lastScreenshot ? (
          <Box
            sx={{
              width: '100%',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              p: 2,
            }}
          >
            <Box
              component="img"
              src={lastScreenshot}
              alt="실시간 화면"
              sx={{
                maxWidth: '100%',
                maxHeight: '100%',
                objectFit: 'contain',
                borderRadius: 2,
                boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
              }}
            />
          </Box>
        ) : getWorkflowDefinition() ? (
          <WorkflowGraph
            definition={getWorkflowDefinition()!}
            executionState={getExecutionState()}
            height="100%"
            onNodeClick={handleNodeClick}
            focusMode={focusMode && isRunning}
            showParameterNode={true}
            parameterConfig={getParameterConfig()}
            onParameterNodeClick={() => setParameterModalOpen(true)}
            isRunning={isRunning}
          />
        ) : (
          <Box
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 2,
            }}
          >
            <AccountTreeIcon sx={{ fontSize: 64, color: 'text.disabled' }} />
            <Typography variant="h6" color="text.secondary">
              워크플로우를 로드하는 중...
            </Typography>
          </Box>
        )}

        {/* VLM 스텝 패널 */}
        {(isRunning || vlmSteps.length > 0) && !showLiveScreen && (
          <VLMStepPanel
            steps={vlmSteps}
            currentNode={currentNode}
            isRunning={isRunning}
            collapsed={stepPanelCollapsed}
            onToggle={() => setStepPanelCollapsed(!stepPanelCollapsed)}
            executionId={executionId}
            workflowId={workflowId}
          />
        )}

        {/* 에이전트 활동 패널 - 항상 표시 (라이브 화면일 때 제외) */}
        {!showLiveScreen && (
          <AgentActivityPanel
            executionId={executionId}
            collapsed={activityPanelCollapsed}
            onToggle={() => setActivityPanelCollapsed(!activityPanelCollapsed)}
          />
        )}
      </Box>

      {/* 파라미터 설정 모달 */}
      <ParameterModal
        open={parameterModalOpen}
        onClose={() => setParameterModalOpen(false)}
        parameters={workflow?.config.parameters || []}
        values={parameters}
        onChange={handleParameterChange}
        onStart={handleStart}
        disabled={isRunning}
      />

      {/* 노드 상세 모달 */}
      <NodeDetailModal
        open={nodeModalOpen}
        onClose={() => setNodeModalOpen(false)}
        executionId={executionId}
        nodeId={selectedNodeId}
        nodeName={workflow?.nodes.find((n) => n.id === selectedNodeId)?.name}
        nodeType={workflow?.nodes.find((n) => n.id === selectedNodeId)?.type}
        instruction={
          workflow?.nodes.find((n) => n.id === selectedNodeId)?.instruction
        }
        metadata={
          workflow?.nodes.find((n) => n.id === selectedNodeId)?.metadata
        }
      />

      {/* 상품 목록 모달 */}
      <ProductListModal
        open={productModalOpen}
        onClose={() => setProductModalOpen(false)}
        initialKeyword={parameters?.keyword as string}
      />
    </Box>
  );
};

export default WorkflowDetail;
