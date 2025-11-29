/**
 * Turbo Flow 스타일 워크플로우 노드 컴포넌트
 * - 클릭 시 상세 정보 표시 지원
 * - VLM 노드의 경우 스크린샷, 액션, 생각 표시
 */

import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Box, Typography, keyframes, Chip } from '@mui/material';
import {
  FiPlay,
  FiCheck,
  FiX,
  FiSearch,
  FiEye,
  FiArrowRight,
  FiFlag,
  FiAlertTriangle,
  FiMonitor,
  FiCpu,
  FiZap,
  FiRefreshCw,
  FiLink,
} from 'react-icons/fi';

// 실행 중 그라데이션 애니메이션
const gradientFlow = keyframes`
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
`;

const pulse = keyframes`
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
`;

const spin = keyframes`
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
`;

export interface WorkflowNodeData {
  label: string;
  description: string;
  status: 'pending' | 'running' | 'success' | 'failed' | 'skipped';
  nodeType?: 'start' | 'process' | 'condition' | 'end' | 'error' | 'vlm';
  executionTime?: number; // 실행 시간 (ms)
  stepCount?: number; // VLM 스텝 수
  instruction?: string; // VLM 에이전트에 전달하는 명령 (시스템 프롬프트)
  onClick?: () => void;
  // 재사용/메모리 설정
  reusable?: boolean;  // trace 재사용 가능 여부
  reuseTrace?: boolean;  // 이전 trace 재사용 여부
  shareMemory?: boolean;  // 메모리 공유 여부
  cacheKeyParams?: string[];  // 캐시 키 파라미터
}

// 노드 타입별 설정
const nodeTypeConfig: Record<string, { icon: React.ElementType; color: string; gradient: string }> = {
  start: {
    icon: FiPlay,
    color: '#10b981',
    gradient: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
  },
  process: {
    icon: FiCpu,
    color: '#6366f1',
    gradient: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)',
  },
  vlm: {
    icon: FiMonitor,
    color: '#f59e0b',
    gradient: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
  },
  condition: {
    icon: FiEye,
    color: '#8b5cf6',
    gradient: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)',
  },
  end: {
    icon: FiFlag,
    color: '#06b6d4',
    gradient: 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)',
  },
  error: {
    icon: FiAlertTriangle,
    color: '#ef4444',
    gradient: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
  },
  default: {
    icon: FiZap,
    color: '#64748b',
    gradient: 'linear-gradient(135deg, #64748b 0%, #475569 100%)',
  },
};

// 상태별 테두리 색상
const statusBorderColors = {
  pending: '#e2e8f0',
  running: '#3b82f6',
  success: '#22c55e',
  failed: '#ef4444',
  skipped: '#94a3b8',
};

const WorkflowNode = memo(({ data, selected }: NodeProps<WorkflowNodeData>) => {
  const nodeType = data.nodeType || 'default';
  const typeConfig = nodeTypeConfig[nodeType] || nodeTypeConfig.default;
  const NodeIcon = typeConfig.icon;

  const isRunning = data.status === 'running';
  const isSuccess = data.status === 'success';
  const isFailed = data.status === 'failed';
  const isVLM = nodeType === 'vlm';
  // VLM 노드는 항상 클릭 가능 (상세정보 보기), 다른 노드는 성공/실패 시에만 클릭 가능
  const isClickable = data.onClick && (isVLM || isSuccess || isFailed);

  const handleClick = () => {
    if (data.onClick && (isVLM || isSuccess || isFailed)) {
      data.onClick();
    }
  };

  return (
    <Box
      onClick={handleClick}
      sx={{
        minWidth: 240,
        maxWidth: 300,
        backgroundColor: '#ffffff',
        borderRadius: '12px',
        border: '2px solid',
        borderColor: isRunning ? 'transparent' : statusBorderColors[data.status],
        boxShadow: selected
          ? '0 0 0 2px rgba(59, 130, 246, 0.5), 0 10px 25px rgba(0, 0, 0, 0.15)'
          : '0 4px 12px rgba(0, 0, 0, 0.08)',
        transition: 'all 0.2s ease',
        position: 'relative',
        overflow: 'hidden',
        cursor: isClickable ? 'pointer' : 'default',
        '&:hover': {
          boxShadow: '0 8px 25px rgba(0, 0, 0, 0.15)',
          transform: isClickable ? 'translateY(-2px)' : 'none',
        },
        // 실행 중일 때 그라데이션 테두리 애니메이션
        ...(isRunning && {
          '&::before': {
            content: '""',
            position: 'absolute',
            top: -2,
            left: -2,
            right: -2,
            bottom: -2,
            background: 'linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899, #3b82f6)',
            backgroundSize: '300% 300%',
            animation: `${gradientFlow} 2s ease infinite`,
            borderRadius: '14px',
            zIndex: -1,
          },
        }),
      }}
    >
      {/* 상단 컬러 바 */}
      <Box
        sx={{
          height: 4,
          background: isSuccess
            ? '#22c55e'
            : isFailed
            ? '#ef4444'
            : isRunning
            ? `linear-gradient(90deg, #3b82f6, #8b5cf6, #3b82f6)`
            : typeConfig.gradient,
          backgroundSize: isRunning ? '200% 200%' : 'auto',
          animation: isRunning ? `${gradientFlow} 1.5s ease infinite` : 'none',
        }}
      />

      {/* 메인 콘텐츠 */}
      <Box sx={{ p: 2 }}>
        {/* 헤더 */}
        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.5, mb: 1.5 }}>
          {/* 아이콘 */}
          <Box
            sx={{
              width: 40,
              height: 40,
              borderRadius: '10px',
              background: isSuccess
                ? '#22c55e'
                : isFailed
                ? '#ef4444'
                : typeConfig.gradient,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
            }}
          >
            {isRunning ? (
              <Box
                sx={{
                  width: 20,
                  height: 20,
                  border: '2.5px solid #ffffff',
                  borderTopColor: 'transparent',
                  borderRadius: '50%',
                  animation: `${spin} 0.8s linear infinite`,
                }}
              />
            ) : isSuccess ? (
              <FiCheck color="#ffffff" size={20} strokeWidth={3} />
            ) : isFailed ? (
              <FiX color="#ffffff" size={20} strokeWidth={3} />
            ) : (
              <NodeIcon color="#ffffff" size={20} />
            )}
          </Box>

          {/* 타이틀 + 상태 */}
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography
              sx={{
                fontSize: '14px',
                fontWeight: 700,
                color: '#1e293b',
                lineHeight: 1.3,
                mb: 0.5,
              }}
            >
              {data.label}
            </Typography>

            {/* 상태 뱃지 */}
            <Chip
              size="small"
              label={
                isRunning ? '실행 중' :
                isSuccess ? '완료' :
                isFailed ? '실패' :
                data.status === 'skipped' ? '건너뜀' : '대기'
              }
              sx={{
                height: 20,
                fontSize: '10px',
                fontWeight: 600,
                backgroundColor: isRunning
                  ? 'rgba(59, 130, 246, 0.1)'
                  : isSuccess
                  ? 'rgba(34, 197, 94, 0.1)'
                  : isFailed
                  ? 'rgba(239, 68, 68, 0.1)'
                  : 'rgba(148, 163, 184, 0.1)',
                color: isRunning
                  ? '#3b82f6'
                  : isSuccess
                  ? '#22c55e'
                  : isFailed
                  ? '#ef4444'
                  : '#64748b',
                animation: isRunning ? `${pulse} 1.5s ease infinite` : 'none',
                '& .MuiChip-label': {
                  px: 1,
                },
              }}
            />
          </Box>
        </Box>

        {/* 설명 */}
        <Typography
          sx={{
            fontSize: '12px',
            color: '#64748b',
            lineHeight: 1.5,
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
          }}
        >
          {data.description}
        </Typography>

        {/* 재사용/메모리 설정 배지 */}
        {(data.reusable || data.reuseTrace || data.shareMemory) && (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
              mt: 1,
              flexWrap: 'wrap',
            }}
          >
            {data.reuseTrace && (
              <Chip
                icon={<FiRefreshCw size={10} />}
                label="Trace 재사용"
                size="small"
                title="이전 성공 trace를 재사용하여 빠르게 실행합니다"
                sx={{
                  height: 18,
                  fontSize: '9px',
                  fontWeight: 600,
                  backgroundColor: '#dbeafe',
                  color: '#1d4ed8',
                  '& .MuiChip-icon': {
                    color: '#1d4ed8',
                    ml: 0.5,
                    mr: -0.5,
                  },
                  '& .MuiChip-label': {
                    px: 0.5,
                  },
                }}
              />
            )}
            {data.shareMemory && (
              <Chip
                icon={<FiLink size={10} />}
                label="메모리 공유"
                size="small"
                title="이전 노드와 컨텍스트를 공유합니다"
                sx={{
                  height: 18,
                  fontSize: '9px',
                  fontWeight: 600,
                  backgroundColor: '#fef3c7',
                  color: '#92400e',
                  '& .MuiChip-icon': {
                    color: '#92400e',
                    ml: 0.5,
                    mr: -0.5,
                  },
                  '& .MuiChip-label': {
                    px: 0.5,
                  },
                }}
              />
            )}
          </Box>
        )}

        {/* 실행 정보 (완료된 경우) 또는 VLM 노드의 경우 항상 표시 */}
        {((isSuccess || isFailed) && (data.executionTime || data.stepCount)) || isVLM ? (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1.5,
              mt: 1.5,
              pt: 1.5,
              borderTop: '1px solid #f1f5f9',
            }}
          >
            {data.executionTime !== undefined && (
              <Typography sx={{ fontSize: '11px', color: '#94a3b8' }}>
                {data.executionTime < 1000
                  ? `${data.executionTime}ms`
                  : `${(data.executionTime / 1000).toFixed(1)}s`}
              </Typography>
            )}
            {data.stepCount !== undefined && (
              <Typography sx={{ fontSize: '11px', color: '#94a3b8' }}>
                {data.stepCount}단계
              </Typography>
            )}
            {isClickable && (
              <Typography
                sx={{
                  fontSize: '11px',
                  color: '#3b82f6',
                  fontWeight: 600,
                  ml: 'auto',
                }}
              >
                클릭하여 상세보기
              </Typography>
            )}
          </Box>
        ) : null}
      </Box>

      {/* 핸들 - 입력 (왼쪽) */}
      <Handle
        type="target"
        position={Position.Left}
        style={{
          width: 12,
          height: 12,
          backgroundColor: '#ffffff',
          border: '2px solid #94a3b8',
          left: -6,
        }}
      />

      {/* 핸들 - 성공 출력 (오른쪽 상단) */}
      <Handle
        type="source"
        position={Position.Right}
        id="success"
        style={{
          width: 12,
          height: 12,
          backgroundColor: '#ffffff',
          border: '2px solid #22c55e',
          right: -6,
          top: '35%',
        }}
      />

      {/* 핸들 - 실패 출력 (오른쪽 하단) */}
      <Handle
        type="source"
        position={Position.Right}
        id="failure"
        style={{
          width: 12,
          height: 12,
          backgroundColor: '#ffffff',
          border: '2px solid #ef4444',
          right: -6,
          top: '65%',
        }}
      />
    </Box>
  );
});

WorkflowNode.displayName = 'WorkflowNode';

export default WorkflowNode;
