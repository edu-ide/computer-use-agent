/**
 * 노드 상세 정보 모달 - VLM 스텝 정보 표시
 */

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  Box,
  Typography,
  IconButton,
  Chip,
  Divider,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Stepper,
  Step,
  StepLabel,
  StepContent,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import {
  FiCheck,
  FiX,
  FiMonitor,
  FiMousePointer,
  FiMessageSquare,
  FiEye,
  FiPackage,
  FiDatabase,
  FiDollarSign,
  FiShoppingBag,
} from 'react-icons/fi';
import { getApiBaseUrl } from '@/config/api';

interface NodeLog {
  step_number: number;
  timestamp: string;
  screenshot?: string;
  action?: string;
  thought?: string;
  observation?: string;
  tool_calls?: Array<{ name: string; args: Record<string, unknown> }>;
}

interface ProductInfo {
  name: string;
  price?: string | number;
  url?: string;
  seller?: string;
  image?: string;
  rating?: string | number;
  reviews?: string | number;
  delivery?: string;
}

interface NodeDetailModalProps {
  open: boolean;
  onClose: () => void;
  executionId: string | null;
  nodeId: string | null;
  nodeName?: string;
  nodeType?: 'start' | 'process' | 'condition' | 'end' | 'error' | 'vlm';
  instruction?: string; // VLM 에이전트 명령 (시스템 프롬프트)
}

const NodeDetailModal: React.FC<NodeDetailModalProps> = ({
  open,
  onClose,
  executionId,
  nodeId,
  nodeName,
  nodeType,
  instruction,
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<NodeLog[]>([]);
  const [status, setStatus] = useState<'success' | 'failed' | null>(null);
  const [nodeError, setNodeError] = useState<string | null>(null);
  const [nodeData, setNodeData] = useState<Record<string, unknown>>({});
  const [workflowData, setWorkflowData] = useState<Record<string, unknown>>({});

  useEffect(() => {
    if (open && executionId && nodeId) {
      fetchNodeLogs();
    }
  }, [open, executionId, nodeId]);

  const fetchNodeLogs = async () => {
    if (!executionId || !nodeId) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${getApiBaseUrl()}/workflows/executions/${executionId}/nodes/${nodeId}/logs`
      );

      if (!response.ok) {
        throw new Error('노드 로그 조회 실패');
      }

      const data = await response.json();
      setLogs(data.logs || []);
      setStatus(data.status);
      setNodeError(data.error);
      setNodeData(data.data || {});
      setWorkflowData(data.workflow_data || {});
    } catch (err) {
      setError(err instanceof Error ? err.message : '알 수 없는 오류');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          backgroundColor: '#ffffff',
          color: '#1e293b',
          borderRadius: 3,
          maxHeight: '85vh',
        },
      }}
    >
      <DialogTitle
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid #e2e8f0',
          pb: 2,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Box
            sx={{
              width: 40,
              height: 40,
              borderRadius: '10px',
              background: status === 'success'
                ? '#22c55e'
                : status === 'failed'
                ? '#ef4444'
                : 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {status === 'success' ? (
              <FiCheck color="#fff" size={20} />
            ) : status === 'failed' ? (
              <FiX color="#fff" size={20} />
            ) : (
              <FiMonitor color="#fff" size={20} />
            )}
          </Box>
          <Box>
            <Typography variant="h6" fontWeight={700} color="#1e293b">
              {nodeName || nodeId}
            </Typography>
            <Typography variant="body2" color="#64748b">
              노드 실행 상세 정보
            </Typography>
          </Box>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {status && (
            <Chip
              label={status === 'success' ? '성공' : '실패'}
              size="small"
              sx={{
                backgroundColor: status === 'success'
                  ? 'rgba(34, 197, 94, 0.2)'
                  : 'rgba(239, 68, 68, 0.2)',
                color: status === 'success' ? '#22c55e' : '#ef4444',
                fontWeight: 600,
              }}
            />
          )}
          <IconButton onClick={onClose} sx={{ color: '#64748b' }}>
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent sx={{ pt: 3 }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
            <CircularProgress sx={{ color: '#3b82f6' }} />
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        ) : (
          <>
            {/* 에러 표시 */}
            {nodeError && (
              <Alert severity="error" sx={{ mb: 3 }}>
                {nodeError}
              </Alert>
            )}

            {/* VLM 노드의 시스템 프롬프트/명령 표시 */}
            {nodeType === 'vlm' && instruction && (
              <Card
                sx={{
                  backgroundColor: '#fefce8',
                  border: '1px solid #fef08a',
                  mb: 3,
                }}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                    <FiMessageSquare size={16} color="#ca8a04" />
                    <Typography variant="subtitle2" sx={{ color: '#854d0e', fontWeight: 700 }}>
                      에이전트 명령 (시스템 프롬프트)
                    </Typography>
                  </Box>
                  <Typography
                    variant="body2"
                    sx={{
                      color: '#713f12',
                      whiteSpace: 'pre-wrap',
                      fontFamily: 'monospace',
                      fontSize: '13px',
                      lineHeight: 1.6,
                      backgroundColor: 'rgba(202, 138, 4, 0.1)',
                      p: 2,
                      borderRadius: 1,
                    }}
                  >
                    {instruction}
                  </Typography>
                </CardContent>
              </Card>
            )}

            {/* 수집된 데이터 표시 */}
            {(workflowData.products || workflowData.collected_products || nodeData.products) && (
              <Card
                sx={{
                  backgroundColor: '#f0fdf4',
                  border: '1px solid #bbf7d0',
                  mb: 3,
                }}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <FiShoppingBag size={18} color="#16a34a" />
                    <Typography variant="subtitle1" sx={{ color: '#166534', fontWeight: 700 }}>
                      수집된 상품 정보
                    </Typography>
                    <Chip
                      label={`${((workflowData.products || workflowData.collected_products || nodeData.products) as ProductInfo[])?.length || 0}개`}
                      size="small"
                      sx={{
                        ml: 'auto',
                        bgcolor: '#22c55e',
                        color: '#fff',
                        fontWeight: 600,
                      }}
                    />
                  </Box>

                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                    {((workflowData.products || workflowData.collected_products || nodeData.products) as ProductInfo[])?.map((product, idx) => (
                      <Card
                        key={idx}
                        variant="outlined"
                        sx={{
                          bgcolor: '#fff',
                          borderColor: '#dcfce7',
                        }}
                      >
                        <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                          <Box sx={{ display: 'flex', gap: 2 }}>
                            {/* 상품 이미지 */}
                            {product.image && (
                              <Box
                                component="img"
                                src={product.image}
                                alt={product.name}
                                sx={{
                                  width: 60,
                                  height: 60,
                                  objectFit: 'cover',
                                  borderRadius: 1,
                                  border: '1px solid #e2e8f0',
                                }}
                              />
                            )}

                            <Box sx={{ flex: 1, minWidth: 0 }}>
                              {/* 상품명 */}
                              <Typography
                                variant="body2"
                                sx={{
                                  fontWeight: 600,
                                  color: '#1e293b',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  whiteSpace: 'nowrap',
                                }}
                              >
                                {product.name}
                              </Typography>

                              {/* 가격 */}
                              {product.price && (
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
                                  <FiDollarSign size={12} color="#16a34a" />
                                  <Typography variant="body2" sx={{ color: '#16a34a', fontWeight: 700 }}>
                                    {typeof product.price === 'number'
                                      ? product.price.toLocaleString() + '원'
                                      : product.price}
                                  </Typography>
                                </Box>
                              )}

                              {/* 판매자 & 배송 */}
                              <Box sx={{ display: 'flex', gap: 2, mt: 0.5 }}>
                                {product.seller && (
                                  <Typography variant="caption" sx={{ color: '#64748b' }}>
                                    판매자: {product.seller}
                                  </Typography>
                                )}
                                {product.delivery && (
                                  <Chip
                                    label={product.delivery}
                                    size="small"
                                    sx={{
                                      height: 18,
                                      fontSize: '10px',
                                      bgcolor: product.delivery.includes('로켓') ? '#dbeafe' : '#f1f5f9',
                                      color: product.delivery.includes('로켓') ? '#1d4ed8' : '#64748b',
                                    }}
                                  />
                                )}
                              </Box>

                              {/* 평점 & 리뷰 */}
                              {(product.rating || product.reviews) && (
                                <Typography variant="caption" sx={{ color: '#94a3b8', mt: 0.5, display: 'block' }}>
                                  {product.rating && `⭐ ${product.rating}`}
                                  {product.reviews && ` (${product.reviews}개 리뷰)`}
                                </Typography>
                              )}
                            </Box>
                          </Box>

                          {/* URL */}
                          {product.url && (
                            <Typography
                              component="a"
                              href={product.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              variant="caption"
                              sx={{
                                color: '#3b82f6',
                                display: 'block',
                                mt: 1,
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                                '&:hover': { textDecoration: 'underline' },
                              }}
                            >
                              {product.url}
                            </Typography>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </Box>
                </CardContent>
              </Card>
            )}

            {/* 기타 데이터 (JSON) */}
            {Object.keys(workflowData).length > 0 &&
              !workflowData.products &&
              !workflowData.collected_products && (
                <Card
                  sx={{
                    backgroundColor: '#eff6ff',
                    border: '1px solid #bfdbfe',
                    mb: 3,
                  }}
                >
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                      <FiDatabase size={16} color="#2563eb" />
                      <Typography variant="subtitle2" sx={{ color: '#1e40af', fontWeight: 700 }}>
                        수집된 데이터
                      </Typography>
                    </Box>
                    <Box
                      component="pre"
                      sx={{
                        bgcolor: 'rgba(37, 99, 235, 0.05)',
                        p: 2,
                        borderRadius: 1,
                        fontSize: '12px',
                        overflow: 'auto',
                        maxHeight: 200,
                        color: '#1e3a8a',
                        fontFamily: 'monospace',
                      }}
                    >
                      {JSON.stringify(workflowData, null, 2)}
                    </Box>
                  </CardContent>
                </Card>
              )}

            {/* 스텝 로그 */}
            {logs.length > 0 ? (
              <Stepper orientation="vertical" activeStep={logs.length}>
                {logs.map((log, index) => (
                  <Step key={index} completed>
                    <StepLabel
                      StepIconComponent={() => (
                        <Box
                          sx={{
                            width: 28,
                            height: 28,
                            borderRadius: '50%',
                            backgroundColor: '#3b82f6',
                            color: '#fff',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: 12,
                            fontWeight: 600,
                          }}
                        >
                          {log.step_number}
                        </Box>
                      )}
                    >
                      <Typography sx={{ color: '#1e293b', fontWeight: 600 }}>
                        {log.step_number}단계
                      </Typography>
                      <Typography variant="caption" sx={{ color: '#64748b' }}>
                        {new Date(log.timestamp).toLocaleTimeString()}
                      </Typography>
                    </StepLabel>
                    <StepContent>
                      <Card
                        sx={{
                          backgroundColor: '#f8fafc',
                          border: '1px solid #e2e8f0',
                          mb: 2,
                        }}
                      >
                        <CardContent>
                          {/* 스크린샷 */}
                          {log.screenshot && (
                            <Box sx={{ mb: 2 }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                <FiMonitor size={14} color="#64748b" />
                                <Typography variant="caption" sx={{ color: '#64748b' }}>
                                  스크린샷
                                </Typography>
                              </Box>
                              <Box
                                component="img"
                                src={log.screenshot.startsWith('data:') ? log.screenshot : `data:image/png;base64,${log.screenshot}`}
                                alt={`${log.step_number}단계 스크린샷`}
                                sx={{
                                  width: '100%',
                                  maxHeight: 300,
                                  objectFit: 'contain',
                                  borderRadius: 1,
                                  border: '1px solid #e2e8f0',
                                }}
                              />
                            </Box>
                          )}

                          {/* 생각 */}
                          {log.thought && (
                            <Box sx={{ mb: 2 }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                <FiMessageSquare size={14} color="#8b5cf6" />
                                <Typography variant="caption" sx={{ color: '#8b5cf6' }}>
                                  생각
                                </Typography>
                              </Box>
                              <Typography
                                variant="body2"
                                sx={{
                                  color: '#1e293b',
                                  backgroundColor: 'rgba(139, 92, 246, 0.08)',
                                  p: 1.5,
                                  borderRadius: 1,
                                  whiteSpace: 'pre-wrap',
                                }}
                              >
                                {log.thought}
                              </Typography>
                            </Box>
                          )}

                          {/* 액션 */}
                          {log.action && (
                            <Box sx={{ mb: 2 }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                <FiMousePointer size={14} color="#3b82f6" />
                                <Typography variant="caption" sx={{ color: '#3b82f6' }}>
                                  액션
                                </Typography>
                              </Box>
                              <Typography
                                variant="body2"
                                sx={{
                                  color: '#1e293b',
                                  backgroundColor: 'rgba(59, 130, 246, 0.08)',
                                  p: 1.5,
                                  borderRadius: 1,
                                  fontFamily: 'monospace',
                                }}
                              >
                                {log.action}
                              </Typography>
                            </Box>
                          )}

                          {/* 관찰 */}
                          {log.observation && (
                            <Box>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                <FiEye size={14} color="#22c55e" />
                                <Typography variant="caption" sx={{ color: '#22c55e' }}>
                                  관찰
                                </Typography>
                              </Box>
                              <Typography
                                variant="body2"
                                sx={{
                                  color: '#1e293b',
                                  backgroundColor: 'rgba(34, 197, 94, 0.08)',
                                  p: 1.5,
                                  borderRadius: 1,
                                  whiteSpace: 'pre-wrap',
                                }}
                              >
                                {log.observation}
                              </Typography>
                            </Box>
                          )}

                          {/* Tool Calls */}
                          {log.tool_calls && log.tool_calls.length > 0 && (
                            <Box sx={{ mt: 2 }}>
                              <Typography variant="caption" sx={{ color: '#64748b' }}>
                                도구 호출:
                              </Typography>
                              {log.tool_calls.map((tc, tcIdx) => (
                                <Chip
                                  key={tcIdx}
                                  label={tc.name}
                                  size="small"
                                  sx={{
                                    ml: 1,
                                    backgroundColor: '#e2e8f0',
                                    color: '#1e293b',
                                  }}
                                />
                              ))}
                            </Box>
                          )}
                        </CardContent>
                      </Card>
                    </StepContent>
                  </Step>
                ))}
              </Stepper>
            ) : (
              <Box sx={{ textAlign: 'center', py: 6 }}>
                <FiMonitor size={48} color="#94a3b8" />
                <Typography variant="body1" sx={{ color: '#64748b', mt: 2 }}>
                  실행 로그가 없습니다
                </Typography>
                <Typography variant="body2" sx={{ color: '#94a3b8', mt: 1 }}>
                  이 노드는 VLM 에이전트를 사용하지 않았거나 로그가 기록되지 않았습니다.
                </Typography>
              </Box>
            )}
          </>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default NodeDetailModal;
