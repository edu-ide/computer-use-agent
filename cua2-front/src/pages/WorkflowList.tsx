/**
 * 워크플로우 목록 페이지 - n8n 스타일 카드 그리드
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardActionArea,
  Grid,
  IconButton,
  Chip,
  CircularProgress,
  Alert,
  Container,
  AppBar,
  Toolbar,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import YouTubeIcon from '@mui/icons-material/YouTube';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import DarkModeOutlined from '@mui/icons-material/DarkModeOutlined';
import LightModeOutlined from '@mui/icons-material/LightModeOutlined';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { selectIsDarkMode, useAgentStore } from '@/stores/agentStore';
import {
  listWorkflows,
  WorkflowDefinition,
} from '@/services/workflowApi';

// 아이콘 매핑
const ICON_MAP: Record<string, React.ElementType> = {
  ShoppingCart: ShoppingCartIcon,
  YouTube: YouTubeIcon,
  AccountTree: AccountTreeIcon,
};

const WorkflowList: React.FC = () => {
  const navigate = useNavigate();
  const isDarkMode = useAgentStore(selectIsDarkMode);
  const toggleDarkMode = useAgentStore((state) => state.toggleDarkMode);

  const [workflows, setWorkflows] = useState<WorkflowDefinition[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadWorkflows();
  }, []);

  const loadWorkflows = async () => {
    setLoading(true);
    try {
      const result = await listWorkflows();
      setWorkflows(result.workflows);
    } catch (err) {
      setError(err instanceof Error ? err.message : '워크플로우 목록 로드 실패');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectWorkflow = (workflowId: string) => {
    navigate(`/workflows/${workflowId}`);
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh', bgcolor: 'background.default' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      {/* 상단 앱바 */}
      <AppBar position="static" color="default" elevation={0} sx={{ borderBottom: '1px solid', borderColor: 'divider' }}>
        <Toolbar>
          <IconButton edge="start" onClick={() => navigate('/')} sx={{ mr: 2 }}>
            <ArrowBackIcon />
          </IconButton>
          <AccountTreeIcon sx={{ mr: 1.5, color: 'primary.main' }} />
          <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 600 }}>
            워크플로우
          </Typography>
          <IconButton onClick={toggleDarkMode}>
            {isDarkMode ? <LightModeOutlined /> : <DarkModeOutlined />}
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* 메인 콘텐츠 */}
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Typography variant="h5" fontWeight={700} sx={{ mb: 1 }}>
          워크플로우 선택
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
          실행할 자동화 워크플로우를 선택하세요
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        <Grid container spacing={3}>
          {workflows.map((workflow) => {
            const IconComponent = ICON_MAP[workflow.config.icon] || AccountTreeIcon;

            return (
              <Grid item xs={12} sm={6} md={4} key={workflow.config.id}>
                <Card
                  sx={{
                    height: '100%',
                    border: '1px solid',
                    borderColor: 'divider',
                    transition: 'all 0.2s',
                    '&:hover': {
                      borderColor: 'primary.main',
                      boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                      transform: 'translateY(-2px)',
                    },
                  }}
                >
                  <CardActionArea
                    onClick={() => handleSelectWorkflow(workflow.config.id)}
                    sx={{ height: '100%' }}
                  >
                    <CardContent sx={{ p: 3 }}>
                      {/* 아이콘 + 배지 */}
                      <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', mb: 2 }}>
                        <Box
                          sx={{
                            width: 56,
                            height: 56,
                            borderRadius: 2,
                            bgcolor: workflow.config.color,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                          }}
                        >
                          <IconComponent sx={{ color: 'white', fontSize: 28 }} />
                        </Box>
                        <Chip
                          label={`${workflow.nodes.length} 노드`}
                          size="small"
                          variant="outlined"
                        />
                      </Box>

                      {/* 제목 */}
                      <Typography variant="h6" fontWeight={600} sx={{ mb: 1 }}>
                        {workflow.config.name}
                      </Typography>

                      {/* 설명 */}
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2, minHeight: 40 }}>
                        {workflow.config.description}
                      </Typography>

                      {/* 파라미터 미리보기 */}
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {workflow.config.parameters.slice(0, 3).map((param) => (
                          <Chip
                            key={param.name}
                            label={param.label}
                            size="small"
                            sx={{ fontSize: '0.7rem' }}
                          />
                        ))}
                        {workflow.config.parameters.length > 3 && (
                          <Chip
                            label={`+${workflow.config.parameters.length - 3}`}
                            size="small"
                            sx={{ fontSize: '0.7rem' }}
                          />
                        )}
                      </Box>

                      {/* 실행 버튼 힌트 */}
                      <Box
                        sx={{
                          mt: 2,
                          pt: 2,
                          borderTop: '1px solid',
                          borderColor: 'divider',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: 1,
                          color: 'primary.main',
                        }}
                      >
                        <PlayArrowIcon fontSize="small" />
                        <Typography variant="body2" fontWeight={600}>
                          클릭하여 실행
                        </Typography>
                      </Box>
                    </CardContent>
                  </CardActionArea>
                </Card>
              </Grid>
            );
          })}
        </Grid>

        {workflows.length === 0 && !error && (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <AccountTreeIcon sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
            <Typography variant="h6" color="text.secondary">
              등록된 워크플로우가 없습니다
            </Typography>
          </Box>
        )}
      </Container>
    </Box>
  );
};

export default WorkflowList;
