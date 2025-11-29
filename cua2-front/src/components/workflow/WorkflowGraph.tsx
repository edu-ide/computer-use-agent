/**
 * 워크플로우 그래프 컴포넌트 - 수평 레이아웃 (왼쪽→오른쪽)
 */

import React, { useCallback, useEffect, useMemo, useRef } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  useNodesState,
  useEdgesState,
  useReactFlow,
  ReactFlowProvider,
  MarkerType,
  ConnectionLineType,
  Position,
} from 'reactflow';
import dagre from 'dagre';
import 'reactflow/dist/style.css';
import { Box } from '@mui/material';

import WorkflowNode, { WorkflowNodeData } from './WorkflowNode';
import WorkflowEdge from './WorkflowEdge';
import ParameterNode, { ParameterNodeData } from './ParameterNode';

// Dagre 레이아웃 설정
const NODE_WIDTH = 280;
const NODE_HEIGHT = 140;

const getLayoutedElements = (
  nodes: Node<WorkflowNodeData>[],
  edges: Edge[],
  direction: 'TB' | 'LR' = 'LR'
) => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));

  const isHorizontal = direction === 'LR';
  dagreGraph.setGraph({
    rankdir: direction,
    nodesep: 80,  // 노드 간 세로 간격
    ranksep: 120, // 레벨 간 가로 간격
    edgesep: 50,  // 엣지 간 간격
    marginx: 50,
    marginy: 50,
  });

  // 노드 추가
  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  });

  // 엣지 추가
  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  // 레이아웃 계산
  dagre.layout(dagreGraph);

  // 계산된 위치 적용
  const layoutedNodes = nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    return {
      ...node,
      position: {
        x: nodeWithPosition.x - NODE_WIDTH / 2,
        y: nodeWithPosition.y - NODE_HEIGHT / 2,
      },
      sourcePosition: isHorizontal ? Position.Right : Position.Bottom,
      targetPosition: isHorizontal ? Position.Left : Position.Top,
    };
  });

  return { nodes: layoutedNodes, edges };
};

// 노드 타입 등록
const nodeTypes = {
  workflow: WorkflowNode,
  parameter: ParameterNode,
};

// 엣지 타입 등록
const edgeTypes = {
  workflow: WorkflowEdge,
};

export interface WorkflowDefinition {
  nodes: Array<{
    id: string;
    name: string;
    description: string;
    status?: string;
    nodeType?: string;
    // 시간 설정
    timeout_sec?: number;
    avg_duration_sec?: number;
    // 재사용/메모리 설정
    reusable?: boolean;
    reuse_trace?: boolean;
    share_memory?: boolean;
    cache_key_params?: string[];
    // 에이전트 정보
    agent_type?: string;  // VLMAgent, SearchAgent, AnalysisAgent 등
    model_id?: string;    // local-qwen3-vl, gpt-4o 등
    // 클릭 가능 여부
    clickable?: boolean;
  }>;
  edges: Array<{
    source: string;
    target: string;
    type: 'success' | 'failure';
  }>;
  startNode: string;
}

export interface ExecutionState {
  currentNode: string | null;
  completedNodes: string[];
  failedNodes: string[];
  // 현재 노드 상세 정보
  currentNodeStartTime?: string;  // 현재 노드 시작 시간 (ISO string)
  currentAction?: string;  // 현재 수행 중인 액션
  currentThought?: string;  // 현재 생각/판단
  currentObservation?: string;  // 현재 관찰 결과
  stepCount?: number;  // 현재 노드의 스텝 수
  // 에러 추적
  consecutiveErrors?: number;  // 연속 에러 횟수
  lastError?: string;  // 마지막 에러 메시지
}

export interface ParameterConfig {
  name: string;
  label: string;
  value: unknown;
}

interface WorkflowGraphProps {
  definition: WorkflowDefinition;
  executionState?: ExecutionState | null;
  height?: number | string;
  onNodeClick?: (nodeId: string) => void;
  focusMode?: boolean; // 실행 중인 노드 자동 포커스
  // 파라미터 노드 관련
  showParameterNode?: boolean;
  parameterConfig?: ParameterConfig[];
  onParameterNodeClick?: () => void;
  isRunning?: boolean;
}

// 파라미터 노드 크기 (일반 노드보다 작음)
const PARAM_NODE_WIDTH = 220;
const PARAM_NODE_HEIGHT = 120;

// 내부 그래프 컴포넌트 (useReactFlow 사용을 위해 분리)
const WorkflowGraphInner: React.FC<WorkflowGraphProps> = ({
  definition,
  executionState,
  height = 500,
  onNodeClick,
  focusMode = false,
  showParameterNode = false,
  parameterConfig = [],
  onParameterNodeClick,
  isRunning = false,
}) => {
  const { fitView, setCenter } = useReactFlow();
  const prevCurrentNodeRef = useRef<string | null>(null);
  // 노드 상태 계산
  const getNodeStatus = useCallback(
    (nodeId: string): WorkflowNodeData['status'] => {
      if (!executionState) return 'pending';
      if (executionState.currentNode === nodeId) return 'running';
      if (executionState.completedNodes.includes(nodeId)) return 'success';
      if (executionState.failedNodes.includes(nodeId)) return 'failed';
      return 'pending';
    },
    [executionState]
  );

  // 노드 타입 결정
  const getNodeType = useCallback(
    (nodeId: string, nodeType?: string): WorkflowNodeData['nodeType'] => {
      if (nodeType) return nodeType as WorkflowNodeData['nodeType'];
      if (nodeId === definition.startNode) return 'start';
      if (nodeId.includes('error') || nodeId.includes('handler')) return 'error';
      if (nodeId.includes('complete') || nodeId.includes('end')) return 'end';
      if (nodeId.includes('check') || nodeId.includes('condition')) return 'condition';
      // VLM 관련 노드
      if (nodeId.includes('analyze') || nodeId.includes('search') || nodeId.includes('open')) return 'vlm';
      return 'process';
    },
    [definition.startNode]
  );

  // 노드 생성 (위치는 나중에 dagre로 계산)
  const rawNodes = useMemo(() => {
    const workflowNodes = definition.nodes.map((nodeDef) => {
      const status = getNodeStatus(nodeDef.id);
      const nodeType = getNodeType(nodeDef.id, nodeDef.nodeType);

      return {
        id: nodeDef.id,
        type: 'workflow',
        position: { x: 0, y: 0 }, // dagre가 계산
        data: {
          label: nodeDef.name,
          description: nodeDef.description,
          status: status,
          nodeType: nodeType,
          onClick: onNodeClick ? () => onNodeClick(nodeDef.id) : undefined,
          // 시간 설정 (snake_case → camelCase)
          timeoutSec: nodeDef.timeout_sec,
          avgDurationSec: nodeDef.avg_duration_sec,
          // 재사용/메모리 설정 (snake_case → camelCase)
          reusable: nodeDef.reusable,
          reuseTrace: nodeDef.reuse_trace,
          shareMemory: nodeDef.share_memory,
          cacheKeyParams: nodeDef.cache_key_params,
          // 에이전트 정보 (snake_case → camelCase)
          agentType: nodeDef.agent_type,
          modelId: nodeDef.model_id,
          // 클릭 가능 여부
          clickable: nodeDef.clickable,
        },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      } as Node<WorkflowNodeData>;
    });

    // 파라미터 노드 추가
    if (showParameterNode) {
      const parameterNode: Node<ParameterNodeData> = {
        id: '__parameters__',
        type: 'parameter',
        position: { x: 0, y: 0 },
        data: {
          parameters: parameterConfig,
          onClick: onParameterNodeClick,
          disabled: isRunning,
        },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      };
      return [parameterNode, ...workflowNodes] as Node[];
    }

    return workflowNodes as Node[];
  }, [definition.nodes, getNodeStatus, getNodeType, onNodeClick, showParameterNode, parameterConfig, onParameterNodeClick, isRunning]);

  // 엣지 생성 (raw)
  const rawEdges = useMemo(() => {
    const workflowEdges = definition.edges.map((edge) => ({
      id: `${edge.source}-${edge.type}-${edge.target}`,
      source: edge.source,
      target: edge.target,
      sourceHandle: edge.type,
      type: 'workflow',
      data: {
        type: edge.type,
        animated: executionState?.currentNode === edge.source,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: edge.type === 'success' ? '#22c55e' : '#ef4444',
        width: 20,
        height: 20,
      },
    }));

    // 파라미터 노드 → 시작 노드 엣지 추가
    if (showParameterNode && definition.startNode) {
      const paramToStartEdge = {
        id: '__parameters__-to-start',
        source: '__parameters__',
        target: definition.startNode,
        sourceHandle: 'output',
        type: 'workflow',
        data: {
          type: 'success' as const,
          animated: false,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: '#64748b',
          width: 20,
          height: 20,
        },
      };
      return [paramToStartEdge, ...workflowEdges];
    }

    return workflowEdges;
  }, [definition.edges, definition.startNode, executionState?.currentNode, showParameterNode]);

  // Dagre 레이아웃 적용
  const { nodes: layoutedNodes, edges: layoutedEdges } = useMemo(() => {
    return getLayoutedElements(rawNodes, rawEdges, 'LR');
  }, [rawNodes, rawEdges]);

  const [nodes, setNodes, onNodesChange] = useNodesState(layoutedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(layoutedEdges);

  // 레이아웃 변경 시 노드/엣지 업데이트
  useEffect(() => {
    setNodes(layoutedNodes);
    setEdges(layoutedEdges);
  }, [layoutedNodes, layoutedEdges, setNodes, setEdges]);

  // 경과 시간 계산 함수
  const calculateElapsedSec = (startTime?: string): number | undefined => {
    if (!startTime) return undefined;
    const start = new Date(startTime).getTime();
    const now = Date.now();
    return Math.floor((now - start) / 1000);
  };

  // 상태 변경 시 노드 업데이트 (위치는 유지)
  useEffect(() => {
    setNodes((nds) =>
      nds.map((node) => {
        const isCurrentNode = executionState?.currentNode === node.id;
        return {
          ...node,
          data: {
            ...node.data,
            status: getNodeStatus(node.id),
            // 현재 실행 중인 노드에만 추가 정보 전달
            currentAction: isCurrentNode ? executionState?.currentAction : undefined,
            currentThought: isCurrentNode ? executionState?.currentThought : undefined,
            currentObservation: isCurrentNode ? executionState?.currentObservation : undefined,
            stepCount: isCurrentNode ? executionState?.stepCount : node.data.stepCount,
            elapsedSec: isCurrentNode ? calculateElapsedSec(executionState?.currentNodeStartTime) : undefined,
            // 에러 추적
            consecutiveErrors: isCurrentNode ? executionState?.consecutiveErrors : undefined,
            lastError: isCurrentNode ? executionState?.lastError : undefined,
          },
        };
      })
    );

    setEdges((eds) =>
      eds.map((edge) => ({
        ...edge,
        data: {
          ...edge.data,
          animated: executionState?.currentNode === edge.source,
        },
      }))
    );
  }, [executionState, setNodes, setEdges, getNodeStatus]);

  // 포커스 모드: 현재 실행 중인 노드로 자동 이동
  useEffect(() => {
    if (!focusMode || !executionState?.currentNode) return;

    // 노드가 바뀌었을 때만 포커스
    if (prevCurrentNodeRef.current === executionState.currentNode) return;
    prevCurrentNodeRef.current = executionState.currentNode;

    // 현재 노드 찾기
    const currentNode = nodes.find(n => n.id === executionState.currentNode);
    if (currentNode) {
      // 노드 중심으로 부드럽게 이동
      setCenter(
        currentNode.position.x + 140, // 노드 중심 (노드 너비 280/2)
        currentNode.position.y + 70,  // 노드 중심 (노드 높이 140/2)
        { zoom: 1, duration: 500 }
      );
    }
  }, [focusMode, executionState?.currentNode, nodes, setCenter]);

  return (
    <Box
      sx={{
        width: '100%',
        height,
        backgroundColor: '#f8fafc',
        borderRadius: 2,
        overflow: 'hidden',
        '.react-flow__node': {
          cursor: 'default',
        },
        '.react-flow__attribution': {
          display: 'none',
        },
      }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        connectionLineType={ConnectionLineType.SmoothStep}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        minZoom={0.3}
        maxZoom={1.5}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={true}
        defaultEdgeOptions={{
          type: 'smoothstep',
        }}
      >
        <Background color="#e2e8f0" gap={25} size={1} />
        <Controls
          showInteractive={false}
          style={{
            backgroundColor: '#ffffff',
            border: '1px solid #e2e8f0',
            borderRadius: 8,
          }}
        />
        <MiniMap
          nodeColor={(node) => {
            const status = (node.data as WorkflowNodeData).status;
            switch (status) {
              case 'running':
                return '#3b82f6';
              case 'success':
                return '#22c55e';
              case 'failed':
                return '#ef4444';
              default:
                return '#94a3b8';
            }
          }}
          maskColor="rgba(255, 255, 255, 0.7)"
          style={{
            backgroundColor: '#ffffff',
            border: '1px solid #e2e8f0',
            borderRadius: 8,
          }}
        />
      </ReactFlow>
    </Box>
  );
};

// 메인 컴포넌트 - ReactFlowProvider로 감싸서 export
const WorkflowGraph: React.FC<WorkflowGraphProps> = (props) => {
  return (
    <ReactFlowProvider>
      <WorkflowGraphInner {...props} />
    </ReactFlowProvider>
  );
};

export default WorkflowGraph;
