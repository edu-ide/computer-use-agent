/**
 * 파라미터 설정 노드 - 워크플로우 시작 전 파라미터 입력용
 */

import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Box, Typography, Chip } from '@mui/material';
import { FiSettings, FiEdit3 } from 'react-icons/fi';

export interface ParameterNodeData {
  parameters: Array<{
    name: string;
    label: string;
    value: unknown;
  }>;
  onClick?: () => void;
  disabled?: boolean;
}

const ParameterNode = memo(({ data, selected }: NodeProps<ParameterNodeData>) => {
  const handleClick = () => {
    if (data.onClick && !data.disabled) {
      data.onClick();
    }
  };

  const displayParams = data.parameters.slice(0, 3);
  const remainingCount = data.parameters.length - 3;

  return (
    <Box
      onClick={handleClick}
      sx={{
        minWidth: 200,
        maxWidth: 260,
        backgroundColor: '#ffffff',
        borderRadius: '12px',
        border: '2px solid',
        borderColor: selected ? '#3b82f6' : '#e2e8f0',
        boxShadow: selected
          ? '0 0 0 2px rgba(59, 130, 246, 0.5), 0 10px 25px rgba(0, 0, 0, 0.15)'
          : '0 4px 12px rgba(0, 0, 0, 0.08)',
        transition: 'all 0.2s ease',
        position: 'relative',
        overflow: 'hidden',
        cursor: data.disabled ? 'not-allowed' : 'pointer',
        opacity: data.disabled ? 0.6 : 1,
        '&:hover': {
          boxShadow: data.disabled ? undefined : '0 8px 25px rgba(0, 0, 0, 0.15)',
          transform: data.disabled ? 'none' : 'translateY(-2px)',
          borderColor: data.disabled ? '#e2e8f0' : '#3b82f6',
        },
      }}
    >
      {/* 상단 컬러 바 */}
      <Box
        sx={{
          height: 4,
          background: 'linear-gradient(135deg, #64748b 0%, #475569 100%)',
        }}
      />

      {/* 메인 콘텐츠 */}
      <Box sx={{ p: 2 }}>
        {/* 헤더 */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1.5 }}>
          <Box
            sx={{
              width: 36,
              height: 36,
              borderRadius: '10px',
              background: 'linear-gradient(135deg, #64748b 0%, #475569 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
            }}
          >
            <FiSettings color="#ffffff" size={18} />
          </Box>

          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography
              sx={{
                fontSize: '13px',
                fontWeight: 700,
                color: '#1e293b',
                lineHeight: 1.3,
              }}
            >
              파라미터 설정
            </Typography>
            <Typography
              sx={{
                fontSize: '11px',
                color: '#64748b',
              }}
            >
              {data.parameters.length}개 항목
            </Typography>
          </Box>

          {!data.disabled && (
            <FiEdit3 color="#3b82f6" size={16} />
          )}
        </Box>

        {/* 파라미터 목록 */}
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
          {displayParams.map((param) => (
            <Box
              key={param.name}
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                backgroundColor: '#f8fafc',
                borderRadius: '6px',
                px: 1,
                py: 0.5,
              }}
            >
              <Typography
                sx={{
                  fontSize: '11px',
                  color: '#64748b',
                  fontWeight: 500,
                }}
              >
                {param.label}
              </Typography>
              <Chip
                size="small"
                label={
                  typeof param.value === 'boolean'
                    ? (param.value ? '예' : '아니오')
                    : String(param.value || '-')
                }
                sx={{
                  height: 18,
                  fontSize: '10px',
                  fontWeight: 600,
                  backgroundColor: '#e2e8f0',
                  color: '#475569',
                  maxWidth: 80,
                  '& .MuiChip-label': {
                    px: 0.75,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  },
                }}
              />
            </Box>
          ))}
          {remainingCount > 0 && (
            <Typography
              sx={{
                fontSize: '10px',
                color: '#94a3b8',
                textAlign: 'center',
                mt: 0.5,
              }}
            >
              +{remainingCount}개 더 보기
            </Typography>
          )}
        </Box>

        {/* 클릭 안내 */}
        {!data.disabled && (
          <Typography
            sx={{
              fontSize: '10px',
              color: '#3b82f6',
              fontWeight: 600,
              textAlign: 'center',
              mt: 1.5,
              pt: 1,
              borderTop: '1px solid #f1f5f9',
            }}
          >
            클릭하여 수정
          </Typography>
        )}
      </Box>

      {/* 핸들 - 출력 (오른쪽) */}
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        style={{
          width: 12,
          height: 12,
          backgroundColor: '#ffffff',
          border: '2px solid #64748b',
          right: -6,
        }}
      />
    </Box>
  );
});

ParameterNode.displayName = 'ParameterNode';

export default ParameterNode;
