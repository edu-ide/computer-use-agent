/**
 * 쿠팡 상품 API 서비스
 */

import { getApiBaseUrl } from '../config/api';

const API_BASE = `${getApiBaseUrl()}/coupang`;

export interface CoupangProduct {
  id: number;
  keyword: string;
  name: string;
  price: number;
  seller_type: string;
  rating: string | null;
  review_count: string | null;
  url: string;
  thumbnail: string | null;
  rank: number;
  timestamp: string;
}

export interface KeywordInfo {
  keyword: string;
  count: number;
  last_collected: string;
}

export interface CoupangStats {
  total_products: number;
  total_keywords: number;
  by_seller_type: Record<string, number>;
}

export interface ChainTask {
  instruction: string;
  on_success: string | null;
  on_failure: string | null;
  status: string;
}

export interface ChainDetail {
  name: string;
  start_task: string;
  tasks: Record<string, ChainTask>;
  status: string;
  is_running: boolean;
}

export interface ChainExecutionState {
  chain_name: string;
  current_task: string | null;
  completed_tasks: string[];
  failed_tasks: string[];
  state: Record<string, unknown>;
  status: string;
  start_time: string | null;
  end_time: string | null;
}

// === 상품 API ===

export async function getProducts(params: {
  keyword?: string;
  seller_type?: string;
  limit?: number;
  offset?: number;
  order_by?: string;
  order_dir?: string;
} = {}): Promise<{ products: CoupangProduct[]; count: number }> {
  const searchParams = new URLSearchParams();

  if (params.keyword) searchParams.set('keyword', params.keyword);
  if (params.seller_type) searchParams.set('seller_type', params.seller_type);
  if (params.limit) searchParams.set('limit', params.limit.toString());
  if (params.offset) searchParams.set('offset', params.offset.toString());
  if (params.order_by) searchParams.set('order_by', params.order_by);
  if (params.order_dir) searchParams.set('order_dir', params.order_dir);

  const response = await fetch(`${API_BASE}/products?${searchParams}`);
  if (!response.ok) throw new Error('상품 조회 실패');
  return response.json();
}

export async function getKeywords(): Promise<{ keywords: KeywordInfo[]; count: number }> {
  const response = await fetch(`${API_BASE}/keywords`);
  if (!response.ok) throw new Error('키워드 조회 실패');
  return response.json();
}

export async function getStats(): Promise<CoupangStats> {
  const response = await fetch(`${API_BASE}/stats`);
  if (!response.ok) throw new Error('통계 조회 실패');
  return response.json();
}

export async function deleteProducts(keyword: string): Promise<{ deleted: number }> {
  const response = await fetch(`${API_BASE}/products/${encodeURIComponent(keyword)}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('상품 삭제 실패');
  return response.json();
}

export async function deleteAllProducts(): Promise<{ deleted: number }> {
  const response = await fetch(`${API_BASE}/products`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('상품 삭제 실패');
  return response.json();
}

// === 체인 API ===

export async function listChains(): Promise<{ chains: string[] }> {
  const response = await fetch(`${API_BASE}/chains`);
  if (!response.ok) throw new Error('체인 목록 조회 실패');
  return response.json();
}

export async function createCoupangChain(
  keyword: string,
  maxPages: number = 5
): Promise<{ chain_name: string; tasks: string[]; start_task: string }> {
  const searchParams = new URLSearchParams();
  searchParams.set('keyword', keyword);
  searchParams.set('max_pages', maxPages.toString());

  const response = await fetch(`${API_BASE}/chains/coupang-collect?${searchParams}`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('체인 생성 실패');
  return response.json();
}

export async function getChainDetail(chainName: string): Promise<ChainDetail> {
  const response = await fetch(`${API_BASE}/chains/${encodeURIComponent(chainName)}`);
  if (!response.ok) throw new Error('체인 상세 조회 실패');
  return response.json();
}

export async function getChainStatus(chainName: string): Promise<ChainExecutionState> {
  const response = await fetch(`${API_BASE}/chains/${encodeURIComponent(chainName)}/status`);
  if (!response.ok) throw new Error('체인 상태 조회 실패');
  return response.json();
}

export async function stopChain(chainName: string): Promise<{ message: string }> {
  const response = await fetch(`${API_BASE}/chains/${encodeURIComponent(chainName)}/stop`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('체인 중지 실패');
  return response.json();
}
