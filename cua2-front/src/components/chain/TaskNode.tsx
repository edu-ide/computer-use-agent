/**
 * 태스크 노드 컴포넌트 - React Flow용
 */

import { Box, Paper, Typography } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import PlayCircleIcon from '@mui/icons-material/PlayCircle';
import { Handle, Position, NodeProps } from 'reactflow';
import { keyframes } from '@mui/system';

const pulse = keyframes`
  0%, 100% {
    box-shadow: 0 0 0 0 rgba(25, 118, 210, 0.4);
  }
  50% {
    box-shadow: 0 0 0 10px rgba(25, 118, 210, 0);
  }
`;

export interface TaskNodeData {
  label: string;
  instruction: string;
  status: 'pending' | 'running' | 'success' | 'failed' | 'skipped';
  onSuccess?: string;
  onFailure?: string;
}

const statusConfig = {
  pending: {
    color: 'grey.400',
    borderColor: 'grey.300',
    icon: HourglassEmptyIcon,
    bgColor: 'grey.50',
  },
  running: {
    color: 'primary.main',
    borderColor: 'primary.main',
    icon: PlayCircleIcon,
    bgColor: 'primary.50',
  },
  success: {
    color: 'success.main',
    borderColor: 'success.main',
    icon: CheckCircleIcon,
    bgColor: 'success.50',
  },
  failed: {
    color: 'error.main',
    borderColor: 'error.main',
    icon: ErrorIcon,
    bgColor: 'error.50',
  },
  skipped: {
    color: 'grey.500',
    borderColor: 'grey.300',
    icon: HourglassEmptyIcon,
    bgColor: 'grey.100',
  },
};

export const TaskNode = ({ data }: NodeProps<TaskNodeData>) => {
  const config = statusConfig[data.status];
  const StatusIcon = config.icon;

  return (
    <Paper
      elevation={data.status === 'running' ? 4 : 1}
      sx={{
        minWidth: 200,
        maxWidth: 280,
        borderRadius: 2,
        border: 2,
        borderColor: config.borderColor,
        borderStyle: data.status === 'skipped' ? 'dashed' : 'solid',
        bgcolor: 'background.paper',
        overflow: 'hidden',
        animation: data.status === 'running' ? `${pulse} 2s infinite` : 'none',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          px: 2,
          py: 1,
          bgcolor: config.bgColor,
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        <StatusIcon
          sx={{
            fontSize: 20,
            color: config.color,
          }}
        />
        <Typography
          variant="subtitle2"
          sx={{
            fontWeight: 600,
            color: config.color,
          }}
        >
          {data.label}
        </Typography>
      </Box>

      {/* Body */}
      <Box sx={{ px: 2, py: 1.5 }}>
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{
            fontSize: '0.75rem',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            display: '-webkit-box',
            WebkitLineClamp: 3,
            WebkitBoxOrient: 'vertical',
          }}
        >
          {data.instruction}
        </Typography>
      </Box>

      {/* Handles */}
      <Handle
        type="target"
        position={Position.Top}
        style={{
          background: '#1976d2',
          width: 10,
          height: 10,
        }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="success"
        style={{
          background: '#4caf50',
          width: 10,
          height: 10,
          left: '30%',
        }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="failure"
        style={{
          background: '#f44336',
          width: 10,
          height: 10,
          left: '70%',
        }}
      />
    </Paper>
  );
};

export default TaskNode;
