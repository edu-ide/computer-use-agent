/**
 * 체인 컨트롤 컴포넌트
 */

import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  TextField,
  Typography,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import { useState } from 'react';
import { ChainExecutionState } from '../../services/coupangApi';

interface ChainControlsProps {
  isRunning: boolean;
  executionState?: ChainExecutionState | null;
  onStart: (keyword: string) => void;
  onStop: () => void;
}

export const ChainControls = ({
  isRunning,
  executionState,
  onStart,
  onStop,
}: ChainControlsProps) => {
  const [keyword, setKeyword] = useState('');

  const handleStart = () => {
    if (keyword.trim()) {
      onStart(keyword.trim());
    }
  };

  const getStatusColor = () => {
    if (!executionState) return 'default';
    switch (executionState.status) {
      case 'running':
        return 'primary';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'stopped':
        return 'warning';
      default:
        return 'default';
    }
  };

  const getStatusLabel = () => {
    if (!executionState) return '대기 중';
    switch (executionState.status) {
      case 'running':
        return '실행 중';
      case 'completed':
        return '완료';
      case 'failed':
        return '실패';
      case 'stopped':
        return '중지됨';
      default:
        return executionState.status;
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          쿠팡 상품 수집
        </Typography>

        {/* 키워드 입력 */}
        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <TextField
            label="검색 키워드"
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            size="small"
            disabled={isRunning}
            sx={{ flex: 1 }}
            placeholder="예: 무선이어폰"
          />
          {!isRunning ? (
            <Button
              variant="contained"
              startIcon={<PlayArrowIcon />}
              onClick={handleStart}
              disabled={!keyword.trim()}
            >
              수집 시작
            </Button>
          ) : (
            <Button
              variant="outlined"
              color="error"
              startIcon={<StopIcon />}
              onClick={onStop}
            >
              중지
            </Button>
          )}
        </Box>

        {/* 상태 표시 */}
        {executionState && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'center' }}>
            <Chip
              label={getStatusLabel()}
              color={getStatusColor()}
              icon={isRunning ? <CircularProgress size={16} color="inherit" /> : undefined}
            />

            {executionState.current_task && (
              <Typography variant="body2" color="text.secondary">
                현재 태스크: <strong>{executionState.current_task}</strong>
              </Typography>
            )}

            <Typography variant="body2" color="text.secondary">
              완료: {executionState.completed_tasks.length}개
            </Typography>

            {executionState.failed_tasks.length > 0 && (
              <Typography variant="body2" color="error">
                실패: {executionState.failed_tasks.length}개
              </Typography>
            )}

            {executionState.state.collected_count !== undefined && (
              <Typography variant="body2" color="success.main">
                수집된 상품: {String(executionState.state.collected_count)}개
              </Typography>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default ChainControls;
