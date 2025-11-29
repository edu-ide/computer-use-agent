"""
쿠팡 상품 관련 API 라우트
"""

from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from cua2_core.services.coupang_db_service import get_coupang_db
from cua2_core.services.task_chain_service import (
    get_chain_service,
    create_coupang_collection_chain,
)

router = APIRouter(prefix="/api/coupang", tags=["coupang"])


@router.get("/products")
async def get_products(
    keyword: Optional[str] = Query(None, description="필터링할 키워드"),
    seller_type: Optional[str] = Query(None, description="배송 타입 필터"),
    limit: int = Query(100, ge=1, le=500, description="최대 결과 수"),
    offset: int = Query(0, ge=0, description="오프셋"),
    order_by: str = Query("timestamp", description="정렬 기준"),
    order_dir: str = Query("DESC", description="정렬 방향"),
):
    """수집된 상품 목록 조회"""
    db = get_coupang_db()
    products = db.get_products(
        keyword=keyword,
        seller_type=seller_type,
        limit=limit,
        offset=offset,
        order_by=order_by,
        order_dir=order_dir,
    )

    return {
        "products": [asdict(p) for p in products],
        "count": len(products),
        "keyword": keyword,
        "limit": limit,
        "offset": offset,
    }


@router.get("/keywords")
async def get_keywords():
    """수집된 키워드 목록 조회"""
    db = get_coupang_db()
    keywords = db.get_keywords()

    return {
        "keywords": keywords,
        "count": len(keywords),
    }


@router.get("/stats")
async def get_stats():
    """수집 통계 조회"""
    db = get_coupang_db()
    stats = db.get_stats()

    return stats


@router.delete("/products/{keyword}")
async def delete_products(keyword: str):
    """특정 키워드의 상품 삭제"""
    db = get_coupang_db()
    deleted = db.delete_products(keyword)

    return {
        "deleted": deleted,
        "keyword": keyword,
    }


@router.delete("/products")
async def delete_all_products():
    """모든 상품 삭제"""
    db = get_coupang_db()
    deleted = db.delete_all_products()

    return {
        "deleted": deleted,
    }


# === 체인 관련 API ===

@router.get("/chains")
async def list_chains():
    """등록된 체인 목록"""
    service = get_chain_service()
    return {
        "chains": service.list_chains(),
    }


@router.post("/chains/coupang-collect")
async def create_coupang_chain(
    keyword: str = Query(..., description="검색 키워드"),
    max_pages: int = Query(5, ge=1, le=20, description="최대 페이지 수"),
):
    """쿠팡 수집 체인 생성"""
    chain = create_coupang_collection_chain(keyword, max_pages)
    service = get_chain_service()
    service.register_chain(chain)

    return {
        "chain_name": chain.name,
        "tasks": list(chain.tasks.keys()),
        "start_task": chain.start_task,
    }


@router.get("/chains/{chain_name}/status")
async def get_chain_status(chain_name: str):
    """체인 실행 상태 조회"""
    service = get_chain_service()
    state = service.get_execution_state(chain_name)

    if not state:
        raise HTTPException(status_code=404, detail="체인 상태를 찾을 수 없음")

    return asdict(state)


@router.post("/chains/{chain_name}/stop")
async def stop_chain(chain_name: str):
    """체인 실행 중지"""
    service = get_chain_service()

    if not service.is_running(chain_name):
        raise HTTPException(status_code=400, detail="체인이 실행 중이 아님")

    service.stop_chain(chain_name)

    return {
        "message": f"체인 중지 요청됨: {chain_name}",
    }


@router.get("/chains/{chain_name}")
async def get_chain_detail(chain_name: str):
    """체인 상세 정보"""
    service = get_chain_service()
    chain = service.get_chain(chain_name)

    if not chain:
        raise HTTPException(status_code=404, detail="체인을 찾을 수 없음")

    return {
        "name": chain.name,
        "start_task": chain.start_task,
        "tasks": {
            name: {
                "instruction": task.instruction,
                "on_success": task.on_success,
                "on_failure": task.on_failure,
                "status": task.status,
            }
            for name, task in chain.tasks.items()
        },
        "status": chain.status,
        "is_running": service.is_running(chain_name),
    }
