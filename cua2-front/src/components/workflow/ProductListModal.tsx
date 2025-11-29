/**
 * 상품 목록 모달 - 워크플로우 실행 중에도 상품 확인 가능
 */

import React, { useEffect, useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Dialog,
  DialogContent,
  DialogTitle,
  FormControl,
  Grid,
  IconButton,
  InputLabel,
  Link,
  MenuItem,
  Paper,
  Select,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import DeleteIcon from '@mui/icons-material/Delete';
import RefreshIcon from '@mui/icons-material/Refresh';
import ViewListIcon from '@mui/icons-material/ViewList';
import ViewModuleIcon from '@mui/icons-material/ViewModule';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import {
  CoupangProduct,
  CoupangStats,
  KeywordInfo,
  deleteProducts,
  getKeywords,
  getProducts,
  getStats,
} from '../../services/coupangApi';

type ViewMode = 'table' | 'grid';

interface ProductListModalProps {
  open: boolean;
  onClose: () => void;
  initialKeyword?: string;
}

const ProductListModal: React.FC<ProductListModalProps> = ({
  open,
  onClose,
  initialKeyword = '',
}) => {
  const [products, setProducts] = useState<CoupangProduct[]>([]);
  const [keywords, setKeywords] = useState<KeywordInfo[]>([]);
  const [stats, setStats] = useState<CoupangStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedKeyword, setSelectedKeyword] = useState<string>(initialKeyword);
  const [viewMode, setViewMode] = useState<ViewMode>('table');
  const [orderBy, setOrderBy] = useState<string>('timestamp');
  const [orderDir, setOrderDir] = useState<string>('DESC');

  const loadData = async () => {
    setLoading(true);
    try {
      const [productsRes, keywordsRes, statsRes] = await Promise.all([
        getProducts({
          keyword: selectedKeyword || undefined,
          limit: 100,
          order_by: orderBy,
          order_dir: orderDir,
        }),
        getKeywords(),
        getStats(),
      ]);
      setProducts(productsRes.products);
      setKeywords(keywordsRes.keywords);
      setStats(statsRes);
    } catch (error) {
      console.error('데이터 로드 실패:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (open) {
      loadData();
    }
  }, [open, selectedKeyword, orderBy, orderDir]);

  useEffect(() => {
    if (initialKeyword) {
      setSelectedKeyword(initialKeyword);
    }
  }, [initialKeyword]);

  const handleDelete = async (keyword: string) => {
    if (!confirm(`"${keyword}" 키워드의 모든 상품을 삭제하시겠습니까?`)) return;
    try {
      await deleteProducts(keyword);
      loadData();
    } catch (error) {
      console.error('삭제 실패:', error);
    }
  };

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('ko-KR').format(price) + '원';
  };

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('ko-KR');
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="xl"
      fullWidth
      PaperProps={{
        sx: {
          height: '90vh',
          maxHeight: '90vh',
        },
      }}
    >
      <DialogTitle
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: 1,
          borderColor: 'divider',
          py: 1.5,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="h6">쿠팡 수집 상품</Typography>
          {stats && (
            <Chip
              label={`${stats.total_products.toLocaleString()}개`}
              size="small"
              color="primary"
            />
          )}
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <IconButton onClick={loadData} disabled={loading} size="small">
            <RefreshIcon />
          </IconButton>
          <IconButton
            component={Link}
            href="/products"
            target="_blank"
            size="small"
            title="새 탭에서 열기"
          >
            <OpenInNewIcon />
          </IconButton>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent sx={{ p: 0, display: 'flex', flexDirection: 'column' }}>
        {/* Stats Cards */}
        {stats && (
          <Box sx={{ p: 2, bgcolor: 'background.default' }}>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                  <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                    <Typography color="text.secondary" variant="body2">
                      총 상품 수
                    </Typography>
                    <Typography variant="h5">
                      {stats.total_products.toLocaleString()}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                  <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                    <Typography color="text.secondary" variant="body2">
                      수집된 키워드
                    </Typography>
                    <Typography variant="h5">
                      {stats.total_keywords.toLocaleString()}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                  <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                    <Typography color="text.secondary" variant="body2">
                      배송 타입별
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 0.5 }}>
                      {Object.entries(stats.by_seller_type).map(([type, count]) => (
                        <Chip
                          key={type}
                          label={`${type}: ${count}`}
                          size="small"
                          variant="outlined"
                        />
                      ))}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        )}

        {/* Filters */}
        <Paper sx={{ p: 2, borderRadius: 0, borderBottom: 1, borderColor: 'divider' }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} sm={4}>
              <FormControl fullWidth size="small">
                <InputLabel>키워드 필터</InputLabel>
                <Select
                  value={selectedKeyword}
                  label="키워드 필터"
                  onChange={(e) => setSelectedKeyword(e.target.value)}
                >
                  <MenuItem value="">전체</MenuItem>
                  {keywords.map((k) => (
                    <MenuItem key={k.keyword} value={k.keyword}>
                      {k.keyword} ({k.count})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={3}>
              <FormControl fullWidth size="small">
                <InputLabel>정렬</InputLabel>
                <Select
                  value={orderBy}
                  label="정렬"
                  onChange={(e) => setOrderBy(e.target.value)}
                >
                  <MenuItem value="timestamp">날짜</MenuItem>
                  <MenuItem value="price">가격</MenuItem>
                  <MenuItem value="name">상품명</MenuItem>
                  <MenuItem value="rank">순위</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={2}>
              <FormControl fullWidth size="small">
                <InputLabel>방향</InputLabel>
                <Select
                  value={orderDir}
                  label="방향"
                  onChange={(e) => setOrderDir(e.target.value)}
                >
                  <MenuItem value="DESC">내림차순</MenuItem>
                  <MenuItem value="ASC">오름차순</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={3} sx={{ display: 'flex', gap: 1 }}>
              <IconButton
                onClick={() => setViewMode('table')}
                color={viewMode === 'table' ? 'primary' : 'default'}
                size="small"
              >
                <ViewListIcon />
              </IconButton>
              <IconButton
                onClick={() => setViewMode('grid')}
                color={viewMode === 'grid' ? 'primary' : 'default'}
                size="small"
              >
                <ViewModuleIcon />
              </IconButton>
              {selectedKeyword && (
                <Button
                  variant="outlined"
                  color="error"
                  size="small"
                  startIcon={<DeleteIcon />}
                  onClick={() => handleDelete(selectedKeyword)}
                >
                  삭제
                </Button>
              )}
            </Grid>
          </Grid>
        </Paper>

        {/* Content Area */}
        <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
          {/* Loading */}
          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress />
            </Box>
          )}

          {/* No Products */}
          {!loading && products.length === 0 && (
            <Paper sx={{ p: 4, textAlign: 'center' }}>
              <Typography color="text.secondary">
                수집된 상품이 없습니다.
              </Typography>
            </Paper>
          )}

          {/* Table View */}
          {!loading && products.length > 0 && viewMode === 'table' && (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell>순위</TableCell>
                    <TableCell>상품명</TableCell>
                    <TableCell align="right">가격</TableCell>
                    <TableCell>배송</TableCell>
                    <TableCell>평점</TableCell>
                    <TableCell>키워드</TableCell>
                    <TableCell>수집일</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {products.map((product) => (
                    <TableRow key={product.id} hover>
                      <TableCell>{product.rank || '-'}</TableCell>
                      <TableCell>
                        <Link
                          href={`https://www.coupang.com${product.url}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          underline="hover"
                        >
                          {product.name.length > 50
                            ? product.name.slice(0, 50) + '...'
                            : product.name}
                        </Link>
                      </TableCell>
                      <TableCell align="right">{formatPrice(product.price)}</TableCell>
                      <TableCell>
                        <Chip
                          label={product.seller_type}
                          size="small"
                          color={product.seller_type === '로켓배송' ? 'primary' : 'default'}
                        />
                      </TableCell>
                      <TableCell>{product.rating || '-'}</TableCell>
                      <TableCell>{product.keyword}</TableCell>
                      <TableCell>{formatDate(product.timestamp)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}

          {/* Grid View */}
          {!loading && products.length > 0 && viewMode === 'grid' && (
            <Grid container spacing={2}>
              {products.map((product) => (
                <Grid item xs={12} sm={6} md={4} lg={3} key={product.id}>
                  <Card variant="outlined">
                    {product.thumbnail && (
                      <Box
                        component="img"
                        src={product.thumbnail}
                        alt={product.name}
                        sx={{
                          width: '100%',
                          height: 150,
                          objectFit: 'cover',
                        }}
                      />
                    )}
                    <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                      <Typography
                        variant="body2"
                        sx={{
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                          mb: 1,
                          minHeight: 40,
                        }}
                      >
                        {product.name}
                      </Typography>
                      <Typography variant="h6" color="primary">
                        {formatPrice(product.price)}
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 0.5, mt: 1, flexWrap: 'wrap' }}>
                        <Chip
                          label={product.seller_type}
                          size="small"
                          color={product.seller_type === '로켓배송' ? 'primary' : 'default'}
                        />
                        {product.rating && (
                          <Chip label={`★ ${product.rating}`} size="small" />
                        )}
                      </Box>
                      <Button
                        component={Link}
                        href={`https://www.coupang.com${product.url}`}
                        target="_blank"
                        size="small"
                        sx={{ mt: 1 }}
                      >
                        쿠팡에서 보기
                      </Button>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}
        </Box>
      </DialogContent>
    </Dialog>
  );
};

export default ProductListModal;
