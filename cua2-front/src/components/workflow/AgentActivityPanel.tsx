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
  TextField,
  InputAdornment,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
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
import SearchIcon from '@mui/icons-material/Search';
import FilterListIcon from '@mui/icons-material/FilterList';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DeleteIcon from '@mui/icons-material/Delete';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import OpenInFullIcon from '@mui/icons-material/OpenInFull';
import CloseFullscreenIcon from '@mui/icons-material/CloseFullscreen';

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
  warning: <ErrorIcon fontSize="inherit" color="warning" />,
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

  // 새로운 상태들
  const [searchQuery, setSearchQuery] = useState('');
  const [filterAgent, setFilterAgent] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false); // 확장 모드
  const [selectedActivity, setSelectedActivity] = useState<AgentActivity | null>(null);
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [filterMenuAnchor, setFilterMenuAnchor] = useState<null | HTMLElement>(null);

  // 필터링된 활동 목록
  const filteredActivities = activities.filter((activity) => {
    const matchesSearch = searchQuery === '' ||
      activity.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (activity.node_id && activity.node_id.toLowerCase().includes(searchQuery.toLowerCase()));
    const matchesFilter = filterAgent === null || activity.agent_type === filterAgent;
    return matchesSearch && matchesFilter;
  });

  // 클립보드 복사
  const handleCopyActivity = (activity: AgentActivity) => {
    const text = `[${activity.timestamp_iso}] [${activity.agent_type}] ${activity.message}${
      activity.details ? '\n상세: ' + JSON.stringify(activity.details, null, 2) : ''
    }`;
    navigator.clipboard.writeText(text);
    setMenuAnchor(null);
  };

  // 전체 로그 복사
  const handleCopyAllLogs = () => {
    const text = filteredActivities
      .map((a) => `[${a.timestamp_iso}] [${a.agent_type}] ${a.message}`)
      .join('\n');
    navigator.clipboard.writeText(text);
    setMenuAnchor(null);
  };

  // 로그 초기화
  const handleClearLogs = () => {
    setActivities([]);
    setMenuAnchor(null);
  };

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
        left: 16,
        width: collapsed ? 200 : expanded ? 500 : 400,
        height: collapsed ? 'auto' : expanded ? '80vh' : 500,
        maxHeight: collapsed ? 'auto' : expanded ? '80vh' : 500,
        zIndex: 10,
        overflow: 'hidden',
        transition: 'all 0.2s ease',
        borderRadius: 2,
        display: 'flex',
        flexDirection: 'column',
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
          flexShrink: 0,
        }}
      >
        <Box
          sx={{ display: 'flex', alignItems: 'center', gap: 1, cursor: 'pointer' }}
          onClick={onToggle}
        >
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
          {filteredActivities.length > 0 && (
            <Chip
              label={filterAgent ? `${filteredActivities.length}/${activities.length}` : activities.length}
              size="small"
              color="primary"
            />
          )}
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          {!collapsed && (
            <>
              <Tooltip title={expanded ? '축소' : '확장'}>
                <IconButton size="small" onClick={() => setExpanded(!expanded)}>
                  {expanded ? <CloseFullscreenIcon fontSize="small" /> : <OpenInFullIcon fontSize="small" />}
                </IconButton>
              </Tooltip>
              <Tooltip title="더보기">
                <IconButton size="small" onClick={(e) => setMenuAnchor(e.currentTarget)}>
                  <MoreVertIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </>
          )}
          <IconButton size="small" onClick={onToggle}>
            {collapsed ? <ExpandMoreIcon /> : <ExpandLessIcon />}
          </IconButton>
        </Box>
      </Box>

      {/* 더보기 메뉴 */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={() => setMenuAnchor(null)}
      >
        <MenuItem onClick={handleCopyAllLogs}>
          <ListItemIcon><ContentCopyIcon fontSize="small" /></ListItemIcon>
          <ListItemText>전체 로그 복사</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleClearLogs}>
          <ListItemIcon><DeleteIcon fontSize="small" /></ListItemIcon>
          <ListItemText>로그 초기화</ListItemText>
        </MenuItem>
      </Menu>

      <Collapse in={!collapsed} sx={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
        {/* 검색 및 필터 */}
        <Box
          sx={{
            px: 1.5,
            py: 1,
            display: 'flex',
            gap: 1,
            alignItems: 'center',
            bgcolor: 'action.hover',
            borderBottom: '1px solid',
            borderColor: 'divider',
            flexShrink: 0,
          }}
        >
          <TextField
            size="small"
            placeholder="검색..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            sx={{
              flex: 1,
              '& .MuiOutlinedInput-root': {
                height: 32,
                fontSize: '0.8rem',
              },
            }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" sx={{ color: 'text.secondary' }} />
                </InputAdornment>
              ),
            }}
          />
          <Tooltip title="필터">
            <IconButton
              size="small"
              onClick={(e) => setFilterMenuAnchor(e.currentTarget)}
              sx={{
                bgcolor: filterAgent ? AGENT_COLORS[filterAgent] : 'transparent',
                color: filterAgent ? 'white' : 'inherit',
                '&:hover': {
                  bgcolor: filterAgent ? AGENT_COLORS[filterAgent] : 'action.hover',
                },
              }}
            >
              <FilterListIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        {/* 필터 메뉴 */}
        <Menu
          anchorEl={filterMenuAnchor}
          open={Boolean(filterMenuAnchor)}
          onClose={() => setFilterMenuAnchor(null)}
        >
          <MenuItem
            onClick={() => { setFilterAgent(null); setFilterMenuAnchor(null); }}
            selected={filterAgent === null}
          >
            <ListItemText>전체 보기</ListItemText>
          </MenuItem>
          <Divider />
          {agents.map((agent) => (
            <MenuItem
              key={agent.type}
              onClick={() => { setFilterAgent(agent.type); setFilterMenuAnchor(null); }}
              selected={filterAgent === agent.type}
            >
              <ListItemIcon sx={{ color: AGENT_COLORS[agent.type] }}>
                {AGENT_ICONS[agent.type]}
              </ListItemIcon>
              <ListItemText>{agent.name}</ListItemText>
            </MenuItem>
          ))}
        </Menu>

        {/* 에이전트 상태 요약 */}
        <Box
          sx={{
            px: 1.5,
            py: 1,
            display: 'flex',
            gap: 0.5,
            flexWrap: 'wrap',
            bgcolor: 'background.paper',
            flexShrink: 0,
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
                onClick={() => setFilterAgent(filterAgent === agent.type ? null : agent.type)}
                sx={{
                  bgcolor: agent.status === 'active' ? AGENT_COLORS[agent.type] : 'action.selected',
                  color: agent.status === 'active' ? 'white' : 'text.secondary',
                  '& .MuiChip-icon': {
                    color: agent.status === 'active' ? 'white' : 'inherit',
                  },
                  transition: 'all 0.2s',
                  cursor: 'pointer',
                  border: filterAgent === agent.type ? '2px solid' : 'none',
                  borderColor: 'primary.main',
                }}
              />
            </Tooltip>
          ))}
        </Box>

        <Divider />

        {/* 활동 로그 */}
        <Box
          sx={{
            flex: 1,
            minHeight: 0, // flex 컨테이너에서 스크롤 허용을 위해 필수
            overflow: 'auto',
            overflowY: 'scroll', // 항상 스크롤바 표시
            // 스크롤바 스타일링 (항상 표시)
            '&::-webkit-scrollbar': {
              width: 8,
            },
            '&::-webkit-scrollbar-track': {
              bgcolor: 'action.hover',
              borderRadius: 4,
            },
            '&::-webkit-scrollbar-thumb': {
              bgcolor: 'primary.light',
              borderRadius: 4,
              border: '2px solid transparent',
              backgroundClip: 'padding-box',
              '&:hover': {
                bgcolor: 'primary.main',
              },
            },
            // Firefox 스크롤바
            scrollbarWidth: 'thin',
            scrollbarColor: 'rgba(25, 118, 210, 0.5) rgba(0, 0, 0, 0.1)',
          }}
        >
          {filteredActivities.length === 0 ? (
            <Box sx={{ p: 3, textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                {activities.length === 0 ? '아직 활동이 없습니다' : '검색 결과가 없습니다'}
              </Typography>
            </Box>
          ) : (
            filteredActivities.map((activity, index) => (
              <Box
                key={activity.id || index}
                sx={{
                  px: 1.5,
                  py: 1,
                  borderBottom: '1px solid',
                  borderColor: 'divider',
                  '&:hover': { bgcolor: 'action.hover' },
                  transition: 'background 0.1s',
                  cursor: 'pointer',
                }}
                onClick={() => setSelectedActivity(selectedActivity?.id === activity.id ? null : activity)}
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
                      width: 22,
                      height: 22,
                      borderRadius: '50%',
                      bgcolor: AGENT_COLORS[activity.agent_type] || '#64748b',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      flexShrink: 0,
                      mt: 0.2,
                    }}
                  >
                    {AGENT_ICONS[activity.agent_type]}
                  </Box>

                  {/* 내용 */}
                  <Box sx={{ flex: 1, minWidth: 0 }}>
                    {/* 메시지 - 확장 모드에서는 줄바꿈 허용 */}
                    <Typography
                      variant="body2"
                      sx={{
                        fontWeight: 500,
                        fontSize: '0.82rem',
                        lineHeight: 1.4,
                        wordBreak: 'break-word',
                        ...(expanded || selectedActivity?.id === activity.id
                          ? { whiteSpace: 'pre-wrap' }
                          : {
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }),
                      }}
                    >
                      {activity.message}
                    </Typography>

                    {/* 메타 정보 */}
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.3, flexWrap: 'wrap' }}>
                      <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                        {activity.time_ago}
                      </Typography>
                      {activity.duration_ms && (
                        <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                          • {activity.duration_ms}ms
                        </Typography>
                      )}
                      {activity.node_id && (
                        <Chip
                          label={activity.node_id}
                          size="small"
                          sx={{ height: 16, fontSize: '0.65rem', bgcolor: 'action.selected' }}
                        />
                      )}
                      {/* 활동 타입 뱃지 */}
                      {activity.activity_type === 'cache_hit' && (
                        <Chip
                          label="캐시"
                          size="small"
                          color="success"
                          sx={{ height: 16, fontSize: '0.65rem' }}
                        />
                      )}
                      {activity.activity_type === 'error' && (
                        <Chip
                          label="오류"
                          size="small"
                          color="error"
                          sx={{ height: 16, fontSize: '0.65rem' }}
                        />
                      )}
                      {activity.activity_type === 'warning' && (
                        <Chip
                          label="주의"
                          size="small"
                          color="warning"
                          sx={{ height: 16, fontSize: '0.65rem' }}
                        />
                      )}
                    </Box>

                    {/* 상세 정보 (선택 시 또는 확장 모드) */}
                    {(selectedActivity?.id === activity.id || (expanded && activity.details && Object.keys(activity.details).length > 0)) && (
                      <Box
                        sx={{
                          mt: 1,
                          p: 1,
                          bgcolor: 'action.hover',
                          borderRadius: 1,
                          fontSize: '0.75rem',
                        }}
                      >
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                          <Typography variant="caption" fontWeight={600}>상세 정보</Typography>
                          <IconButton
                            size="small"
                            onClick={(e) => { e.stopPropagation(); handleCopyActivity(activity); }}
                            sx={{ p: 0.3 }}
                          >
                            <ContentCopyIcon sx={{ fontSize: 14 }} />
                          </IconButton>
                        </Box>
                        <Typography
                          component="pre"
                          sx={{
                            m: 0,
                            fontFamily: 'monospace',
                            fontSize: '0.7rem',
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-all',
                            color: 'text.secondary',
                            maxHeight: 150,
                            overflow: 'auto',
                          }}
                        >
                          {JSON.stringify(activity.details, null, 2)}
                        </Typography>
                      </Box>
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
