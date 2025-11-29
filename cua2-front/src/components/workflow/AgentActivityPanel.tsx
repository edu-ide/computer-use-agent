/**
 * 에이전트 활동 로그 패널 - 실시간 에이전트 작업 표시
 */

import React, { useEffect, useState, useRef } from 'react';
import {
  Box,
  Typography,
  Chip,
  IconButton,
  Collapse,
  Tooltip,
  Paper,
  Divider,
  Badge,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PsychologyIcon from '@mui/icons-material/Psychology';
import VisibilityIcon from '@mui/icons-material/Visibility';
import StorageIcon from '@mui/icons-material/Storage';
import DescriptionIcon from '@mui/icons-material/Description';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import CachedIcon from '@mui/icons-material/Cached';
import CircularProgress from '@mui/material/CircularProgress';

interface AgentActivity {
  id: string;
  agent_type: string;
  activity_type: string;
  message: string;
  details: Record<string, unknown>;
  timestamp: number;
  timestamp_iso: string;
  execution_id: string | null;
  node_id: string | null;
  duration_ms: number | null;
  time_ago: string;
}

interface AgentStatus {
  type: string;
  name: string;
  icon: string;
  color: string;
  status: string;
  latest_activity: AgentActivity | null;
}

interface AgentActivityPanelProps {
  executionId?: string | null;
  collapsed?: boolean;
  onToggle?: () => void;
}

// 에이전트 아이콘 매핑
const AGENT_ICONS: Record<string, React.ReactNode> = {
  orchestrator: <PsychologyIcon fontSize="small" />,
  vlm: <VisibilityIcon fontSize="small" />,
  memory: <StorageIcon fontSize="small" />,
  trace: <DescriptionIcon fontSize="small" />,
};

// 활동 타입 아이콘
const ACTIVITY_ICONS: Record<string, React.ReactNode> = {
  decision: <SmartToyIcon fontSize="inherit" />,
  execution: <CircularProgress size={12} />,
  cache_hit: <CachedIcon fontSize="inherit" color="success" />,
  cache_miss: <CachedIcon fontSize="inherit" color="disabled" />,
  error: <ErrorIcon fontSize="inherit" color="error" />,
  info: <CheckCircleIcon fontSize="inherit" color="info" />,
};

// 에이전트 색상
const AGENT_COLORS: Record<string, string> = {
  orchestrator: '#8B5CF6',
  vlm: '#3B82F6',
  memory: '#10B981',
  trace: '#F59E0B',
};

const AgentActivityPanel: React.FC<AgentActivityPanelProps> = ({
  executionId,
  collapsed = false,
  onToggle,
}) => {
  const [agents, setAgents] = useState<AgentStatus[]>([]);
  const [activities, setActivities] = useState<AgentActivity[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const activitiesEndRef = useRef<HTMLDivElement>(null);

  // WebSocket 연결
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname;
    const port = import.meta.env.VITE_API_PORT || '8000';

    let wsUrl: string;
    if (import.meta.env.PROD) {
      wsUrl = executionId
        ? `${protocol}//${window.location.host}/api/agents/ws/activities/${executionId}`
        : `${protocol}//${window.location.host}/api/agents/ws/activities`;
    } else {
      wsUrl = executionId
        ? `${protocol}//${host}:${port}/api/agents/ws/activities/${executionId}`
        : `${protocol}//${host}:${port}/api/agents/ws/activities`;
    }

    console.log('[AgentActivityPanel] Connecting to:', wsUrl);

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('[AgentActivityPanel] Connected');
      setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('[AgentActivityPanel] Message:', data.type);

        switch (data.type) {
          case 'init':
            setAgents(data.agents || []);
            setActivities(data.recent_activities || data.activities || []);
            break;

          case 'activity':
            setActivities((prev) => [data.activity, ...prev].slice(0, 50));
            // 에이전트 상태 업데이트
            setAgents((prev) =>
              prev.map((agent) =>
                agent.type === data.activity.agent_type
                  ? { ...agent, latest_activity: data.activity, status: 'active' }
                  : agent
              )
            );
            break;

          case 'status_update':
            setAgents(data.agents || []);
            break;
        }
      } catch (err) {
        console.error('[AgentActivityPanel] Parse error:', err);
      }
    };

    ws.onerror = (err) => {
      console.error('[AgentActivityPanel] Error:', err);
    };

    ws.onclose = () => {
      console.log('[AgentActivityPanel] Disconnected');
      setConnected(false);
    };

    wsRef.current = ws;

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [executionId]);

  // 새 활동 시 스크롤
  useEffect(() => {
    if (!collapsed && activitiesEndRef.current) {
      activitiesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [activities.length, collapsed]);

  return (
    <Paper
      elevation={3}
      sx={{
        position: 'absolute',
        top: 16,
        right: 16,
        width: collapsed ? 200 : 360,
        maxHeight: collapsed ? 'auto' : 400,
        zIndex: 10,
        overflow: 'hidden',
        transition: 'all 0.2s ease',
        borderRadius: 2,
      }}
    >
      {/* 헤더 */}
      <Box
        sx={{
          px: 2,
          py: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          bgcolor: 'background.paper',
          borderBottom: collapsed ? 'none' : '1px solid',
          borderColor: 'divider',
          cursor: 'pointer',
        }}
        onClick={onToggle}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Badge
            color={connected ? 'success' : 'error'}
            variant="dot"
            overlap="circular"
          >
            <SmartToyIcon color="primary" fontSize="small" />
          </Badge>
          <Typography variant="subtitle2" fontWeight={600}>
            에이전트 활동
          </Typography>
          {activities.length > 0 && (
            <Chip label={activities.length} size="small" color="primary" />
          )}
        </Box>
        <IconButton size="small">
          {collapsed ? <ExpandMoreIcon /> : <ExpandLessIcon />}
        </IconButton>
      </Box>

      <Collapse in={!collapsed}>
        {/* 에이전트 상태 요약 */}
        <Box
          sx={{
            px: 2,
            py: 1.5,
            display: 'flex',
            gap: 1,
            flexWrap: 'wrap',
            bgcolor: 'action.hover',
          }}
        >
          {agents.map((agent) => (
            <Tooltip
              key={agent.type}
              title={
                agent.latest_activity
                  ? `${agent.latest_activity.message} (${agent.latest_activity.time_ago})`
                  : '대기 중'
              }
              arrow
            >
              <Chip
                icon={AGENT_ICONS[agent.type] as React.ReactElement}
                label={agent.name.split(' ')[0]}
                size="small"
                sx={{
                  bgcolor: agent.status === 'active' ? AGENT_COLORS[agent.type] : 'action.selected',
                  color: agent.status === 'active' ? 'white' : 'text.secondary',
                  '& .MuiChip-icon': {
                    color: agent.status === 'active' ? 'white' : 'inherit',
                  },
                  transition: 'all 0.2s',
                }}
              />
            </Tooltip>
          ))}
        </Box>

        <Divider />

        {/* 활동 로그 */}
        <Box
          sx={{
            maxHeight: 280,
            overflow: 'auto',
            '&::-webkit-scrollbar': { width: 6 },
            '&::-webkit-scrollbar-thumb': {
              bgcolor: 'divider',
              borderRadius: 3,
            },
          }}
        >
          {activities.length === 0 ? (
            <Box sx={{ p: 3, textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                아직 활동이 없습니다
              </Typography>
            </Box>
          ) : (
            activities.map((activity, index) => (
              <Box
                key={activity.id || index}
                sx={{
                  px: 2,
                  py: 1,
                  borderBottom: '1px solid',
                  borderColor: 'divider',
                  '&:hover': { bgcolor: 'action.hover' },
                  transition: 'background 0.1s',
                }}
              >
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 1,
                  }}
                >
                  {/* 에이전트 아이콘 */}
                  <Box
                    sx={{
                      width: 24,
                      height: 24,
                      borderRadius: '50%',
                      bgcolor: AGENT_COLORS[activity.agent_type] || '#64748b',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      flexShrink: 0,
                      mt: 0.3,
                    }}
                  >
                    {AGENT_ICONS[activity.agent_type]}
                  </Box>

                  {/* 내용 */}
                  <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Box
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                      }}
                    >
                      <Typography
                        variant="body2"
                        sx={{
                          fontWeight: 500,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {activity.message}
                      </Typography>
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{ flexShrink: 0, ml: 1 }}
                      >
                        {activity.time_ago}
                      </Typography>
                    </Box>

                    {/* 상세 정보 */}
                    {activity.duration_ms && (
                      <Typography variant="caption" color="text.secondary">
                        {activity.duration_ms}ms
                        {activity.node_id && ` | ${activity.node_id}`}
                      </Typography>
                    )}

                    {/* 활동 타입 뱃지 */}
                    {activity.activity_type === 'cache_hit' && (
                      <Chip
                        label="캐시 히트"
                        size="small"
                        color="success"
                        variant="outlined"
                        sx={{ mt: 0.5, height: 20, fontSize: '0.7rem' }}
                      />
                    )}
                    {activity.activity_type === 'error' && (
                      <Chip
                        label="오류"
                        size="small"
                        color="error"
                        variant="outlined"
                        sx={{ mt: 0.5, height: 20, fontSize: '0.7rem' }}
                      />
                    )}
                  </Box>
                </Box>
              </Box>
            ))
          )}
          <div ref={activitiesEndRef} />
        </Box>
      </Collapse>
    </Paper>
  );
};

export default AgentActivityPanel;
