/**
 * 쿠팡 수집 상품 목록 페이지
 */

import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
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
  TextField,
  Toolbar,
  Typography,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import RefreshIcon from '@mui/icons-material/Refresh';
import HomeIcon from '@mui/icons-material/Home';
import ViewListIcon from '@mui/icons-material/ViewList';
import ViewModuleIcon from '@mui/icons-material/ViewModule';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  CoupangProduct,
  CoupangStats,
  KeywordInfo,
  deleteProducts,
  getKeywords,
  getProducts,
  getStats,
} from '../services/coupangApi';

type ViewMode = 'table' | 'grid';

const Products = () => {
  const navigate = useNavigate();
  const [products, setProducts] = useState<CoupangProduct[]>([]);
  const [keywords, setKeywords] = useState<KeywordInfo[]>([]);
  const [stats, setStats] = useState<CoupangStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedKeyword, setSelectedKeyword] = useState<string>('');
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
    loadData();
  }, [selectedKeyword, orderBy, orderDir]);

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
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      {/* Header */}
      <Paper elevation={0} sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Toolbar>
          <IconButton onClick={() => navigate('/')} sx={{ mr: 2 }}>
            <HomeIcon />
          </IconButton>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            쿠팡 수집 상품
          </Typography>
          <IconButton onClick={loadData} disabled={loading}>
            <RefreshIcon />
          </IconButton>
        </Toolbar>
      </Paper>

      <Container maxWidth="xl" sx={{ py: 3 }}>
        {/* Stats Cards */}
        {stats && (
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={4}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    총 상품 수
                  </Typography>
                  <Typography variant="h4">
                    {stats.total_products.toLocaleString()}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    수집된 키워드
                  </Typography>
                  <Typography variant="h4">
                    {stats.total_keywords.toLocaleString()}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    배송 타입별
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 1 }}>
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
        )}

        {/* Filters */}
        <Paper sx={{ p: 2, mb: 3 }}>
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
              >
                <ViewListIcon />
              </IconButton>
              <IconButton
                onClick={() => setViewMode('grid')}
                color={viewMode === 'grid' ? 'primary' : 'default'}
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

        {/* Loading */}
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress />
          </Box>
        )}

        {/* Products List */}
        {!loading && products.length === 0 && (
          <Paper sx={{ p: 4, textAlign: 'center' }}>
            <Typography color="text.secondary">
              수집된 상품이 없습니다.
            </Typography>
          </Paper>
        )}

        {!loading && products.length > 0 && viewMode === 'table' && (
          <TableContainer component={Paper}>
            <Table size="small">
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
                        color={product.seller_type === '일반배송' ? 'success' : 'default'}
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

        {!loading && products.length > 0 && viewMode === 'grid' && (
          <Grid container spacing={2}>
            {products.map((product) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={product.id}>
                <Card>
                  {product.thumbnail && (
                    <Box
                      component="img"
                      src={product.thumbnail}
                      alt={product.name}
                      sx={{
                        width: '100%',
                        height: 200,
                        objectFit: 'cover',
                      }}
                    />
                  )}
                  <CardContent>
                    <Typography
                      variant="body2"
                      sx={{
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        mb: 1,
                      }}
                    >
                      {product.name}
                    </Typography>
                    <Typography variant="h6" color="primary">
                      {formatPrice(product.price)}
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                      <Chip
                        label={product.seller_type}
                        size="small"
                        color={product.seller_type === '일반배송' ? 'success' : 'default'}
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
      </Container>
    </Box>
  );
};

export default Products;
