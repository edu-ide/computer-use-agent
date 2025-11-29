/**
 * 체인 그래프 컴포넌트 - React Flow 기반
 */

import { Box, Paper, Typography } from '@mui/material';
import { useCallback, useEffect, useMemo, useState } from 'react';
import ReactFlow, {
  Background,
  Controls,
  Edge,
  MarkerType,
  Node,
  useEdgesState,
  useNodesState,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { ChainDetail, ChainExecutionState } from '../../services/coupangApi';
import TaskNode, { TaskNodeData } from './TaskNode';

const nodeTypes = {
  taskNode: TaskNode,
};

interface ChainGraphProps {
  chain: ChainDetail;
  executionState?: ChainExecutionState | null;
}

export const ChainGraph = ({ chain, executionState }: ChainGraphProps) => {
  // 태스크를 노드로 변환
  const initialNodes = useMemo(() => {
    const taskNames = Object.keys(chain.tasks);
    const nodes: Node<TaskNodeData>[] = [];

    // 레이아웃 계산 (간단한 수직 배치)
    const startIndex = taskNames.indexOf(chain.start_task);
    const orderedTasks = [
      chain.start_task,
      ...taskNames.filter((t) => t !== chain.start_task),
    ];

    orderedTasks.forEach((taskName, index) => {
      const task = chain.tasks[taskName];
      const isCompleted = executionState?.completed_tasks.includes(taskName);
      const isFailed = executionState?.failed_tasks.includes(taskName);
      const isRunning = executionState?.current_task === taskName;

      let status: TaskNodeData['status'] = 'pending';
      if (isRunning) status = 'running';
      else if (isCompleted) status = 'success';
      else if (isFailed) status = 'failed';

      nodes.push({
        id: taskName,
        type: 'taskNode',
        position: {
          x: 100 + (index % 3) * 300,
          y: 50 + Math.floor(index / 3) * 180,
        },
        data: {
          label: taskName,
          instruction: task.instruction,
          status,
          onSuccess: task.on_success || undefined,
          onFailure: task.on_failure || undefined,
        },
      });
    });

    return nodes;
  }, [chain, executionState]);

  // 연결선(엣지) 생성
  const initialEdges = useMemo(() => {
    const edges: Edge[] = [];

    Object.entries(chain.tasks).forEach(([taskName, task]) => {
      if (task.on_success) {
        edges.push({
          id: `${taskName}-success-${task.on_success}`,
          source: taskName,
          target: task.on_success,
          sourceHandle: 'success',
          style: { stroke: '#4caf50', strokeWidth: 2 },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: '#4caf50',
          },
          label: 'success',
          labelStyle: { fill: '#4caf50', fontSize: 10 },
        });
      }

      if (task.on_failure) {
        edges.push({
          id: `${taskName}-failure-${task.on_failure}`,
          source: taskName,
          target: task.on_failure,
          sourceHandle: 'failure',
          style: { stroke: '#f44336', strokeWidth: 2, strokeDasharray: '5,5' },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: '#f44336',
          },
          label: 'failure',
          labelStyle: { fill: '#f44336', fontSize: 10 },
        });
      }
    });

    return edges;
  }, [chain]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // executionState가 변경되면 노드 상태 업데이트
  useEffect(() => {
    setNodes((nds) =>
      nds.map((node) => {
        const isCompleted = executionState?.completed_tasks.includes(node.id);
        const isFailed = executionState?.failed_tasks.includes(node.id);
        const isRunning = executionState?.current_task === node.id;

        let status: TaskNodeData['status'] = 'pending';
        if (isRunning) status = 'running';
        else if (isCompleted) status = 'success';
        else if (isFailed) status = 'failed';

        return {
          ...node,
          data: {
            ...node.data,
            status,
          },
        };
      })
    );
  }, [executionState, setNodes]);

  return (
    <Paper sx={{ height: 500, width: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
      >
        <Background />
        <Controls />
      </ReactFlow>
    </Paper>
  );
};

export default ChainGraph;
