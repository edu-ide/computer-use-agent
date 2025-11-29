"""
쿠팡 수집용 에이전트 도구들
VLM 에이전트가 사용할 수 있는 도구 모음
"""

from datetime import datetime
from typing import Optional

from smolagents import tool

from cua2_core.models.coupang_models import CoupangProduct
from cua2_core.services.coupang_db_service import get_coupang_db


# 현재 검색 중인 키워드 (상태 공유용)
_current_keyword: Optional[str] = None
_collected_urls: set[str] = set()


def set_current_keyword(keyword: str):
    """현재 키워드 설정"""
    global _current_keyword, _collected_urls
    _current_keyword = keyword
    _collected_urls = set()


def get_current_keyword() -> Optional[str]:
    """현재 키워드 반환"""
    return _current_keyword


@tool
def save_non_rocket_product(
    name: str,
    price: str,
    url: str,
    seller_type: str = "일반배송",
    rating: str = "",
    review_count: str = "",
    rank: int = 0
) -> str:
    """
    비로켓배송 상품을 DB에 저장합니다.

    Args:
        name: 상품명
        price: 가격 (숫자만, 예: "15900")
        url: 상품 URL
        seller_type: 배송 타입 (일반배송, 로켓직구, 판매자로켓)
        rating: 평점 (예: "4.5")
        review_count: 리뷰 수 (예: "1,234")
        rank: 검색 결과 순위

    Returns:
        저장 결과 메시지
    """
    global _collected_urls

    if not _current_keyword:
        return "Error: 현재 키워드가 설정되지 않았습니다. 먼저 검색을 시작하세요."

    # URL 중복 체크
    if url in _collected_urls:
        return f"Skip: 이미 수집된 상품입니다 - {name[:30]}"

    # 가격 파싱
    try:
        price_int = int(price.replace(",", "").replace("원", "").strip())
    except ValueError:
        price_int = 0

    product = CoupangProduct(
        keyword=_current_keyword,
        name=name,
        price=price_int,
        seller_type=seller_type,
        url=url,
        rating=rating if rating else None,
        review_count=review_count if review_count else None,
        rank=rank,
        timestamp=datetime.utcnow().isoformat(),
    )

    db = get_coupang_db()
    success = db.save_product(product)

    if success:
        _collected_urls.add(url)
        return f"Saved: {name[:30]}... (가격: {price_int:,}원, 타입: {seller_type})"
    else:
        return f"Skip: 중복 상품 - {name[:30]}"


@tool
def get_collected_count() -> str:
    """
    현재 키워드로 수집된 상품 수를 반환합니다.

    Returns:
        수집된 상품 수 정보
    """
    if not _current_keyword:
        return "현재 키워드가 설정되지 않았습니다."

    db = get_coupang_db()
    count = db.get_product_count(_current_keyword)
    total = db.get_product_count()

    return f"현재 키워드 '{_current_keyword}': {count}개 수집, 전체: {total}개"


@tool
def mark_keyword_done() -> str:
    """
    현재 키워드의 수집 완료를 표시합니다.

    Returns:
        완료 메시지
    """
    if not _current_keyword:
        return "현재 키워드가 설정되지 않았습니다."

    db = get_coupang_db()
    count = db.get_product_count(_current_keyword)
    db.mark_keyword_collected(_current_keyword, count)

    return f"키워드 '{_current_keyword}' 수집 완료 ({count}개 상품)"


@tool
def set_search_keyword(keyword: str) -> str:
    """
    새로운 검색 키워드를 설정합니다.

    Args:
        keyword: 검색할 키워드

    Returns:
        설정 결과 메시지
    """
    set_current_keyword(keyword)
    return f"키워드 설정: '{keyword}'"


@tool
def get_collection_stats() -> str:
    """
    전체 수집 통계를 반환합니다.

    Returns:
        통계 정보 문자열
    """
    db = get_coupang_db()
    stats = db.get_stats()

    result = f"=== 수집 통계 ===\n"
    result += f"총 상품: {stats['total_products']}개\n"
    result += f"총 키워드: {stats['total_keywords']}개\n"
    result += f"\n배송 타입별:\n"

    for seller_type, count in stats.get('by_seller_type', {}).items():
        result += f"  - {seller_type}: {count}개\n"

    return result


def get_coupang_tools() -> list:
    """쿠팡 관련 도구 목록 반환"""
    return [
        save_non_rocket_product,
        get_collected_count,
        mark_keyword_done,
        set_search_keyword,
        get_collection_stats,
    ]
