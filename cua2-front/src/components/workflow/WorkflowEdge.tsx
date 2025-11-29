/**
 * 워크플로우 엣지 컴포넌트 - 수평 레이아웃용 (화이트 테마)
 */

import React from 'react';
import {
  BaseEdge,
  EdgeProps,
  getSmoothStepPath,
} from 'reactflow';

interface WorkflowEdgeData {
  type: 'success' | 'failure';
  animated?: boolean;
}

const WorkflowEdge = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  style = {},
  markerEnd,
}: EdgeProps<WorkflowEdgeData>) => {
  const isSuccess = data?.type === 'success';
  const isFailure = data?.type === 'failure';
  const isAnimated = data?.animated;

  // 오프셋 추가 - 성공/실패 엣지가 겹치지 않도록
  const offset = isFailure ? 20 : 0;

  const [edgePath] = getSmoothStepPath({
    sourceX,
    sourceY: sourceY + offset,
    sourcePosition,
    targetX,
    targetY: targetY + offset,
    targetPosition,
    borderRadius: 20,
    offset: 30, // 꺾이는 거리
  });

  const edgeColor = isSuccess ? '#22c55e' : isFailure ? '#ef4444' : '#94a3b8';

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          stroke: edgeColor,
          strokeWidth: 2,
          strokeDasharray: isFailure ? '8 4' : 'none',
          ...style,
        }}
        markerEnd={markerEnd}
      />

      {/* 애니메이션 점 (실행 중일 때) */}
      {isAnimated && (
        <circle r="5" fill={edgeColor}>
          <animateMotion dur="1s" repeatCount="indefinite" path={edgePath} />
        </circle>
      )}
    </>
  );
};

export default WorkflowEdge;
