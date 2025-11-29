"""
쿠팡 상품 DB 서비스 - SQLite 기반
"""

import os
import sqlite3
import threading
from datetime import datetime
from typing import Optional

from cua2_core.models.coupang_models import CoupangProduct


class CoupangDBService:
    """쿠팡 상품 DB 관리 서비스"""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_dir = os.path.expanduser("~/.cua-coupang")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "products.db")

        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """DB 초기화 및 테이블 생성"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS coupang_products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    keyword TEXT NOT NULL,
                    name TEXT NOT NULL,
                    price INTEGER NOT NULL,
                    seller_type TEXT NOT NULL,
                    rating TEXT,
                    review_count TEXT,
                    url TEXT NOT NULL,
                    thumbnail TEXT,
                    rank INTEGER DEFAULT 0,
                    timestamp TEXT NOT NULL,
                    UNIQUE(keyword, url)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_keyword ON coupang_products(keyword)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_seller_type ON coupang_products(seller_type)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collected_keywords (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    keyword TEXT UNIQUE NOT NULL,
                    product_count INTEGER DEFAULT 0,
                    last_collected TEXT,
                    status TEXT DEFAULT 'pending'
                )
            """)
            conn.commit()

    def save_product(self, product: CoupangProduct) -> bool:
        """단일 상품 저장 (중복 시 무시)"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR IGNORE INTO coupang_products
                        (keyword, name, price, seller_type, rating, review_count, url, thumbnail, rank, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        product.keyword,
                        product.name,
                        product.price,
                        product.seller_type,
                        product.rating,
                        product.review_count,
                        product.url,
                        product.thumbnail,
                        product.rank,
                        product.timestamp,
                    ))
                    conn.commit()
                    return conn.total_changes > 0
            except Exception as e:
                print(f"상품 저장 오류: {e}")
                return False

    def save_products(self, products: list[CoupangProduct]) -> int:
        """여러 상품 일괄 저장"""
        saved = 0
        for product in products:
            if self.save_product(product):
                saved += 1
        return saved

    def get_products(
        self,
        keyword: Optional[str] = None,
        seller_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "timestamp",
        order_dir: str = "DESC"
    ) -> list[CoupangProduct]:
        """상품 조회"""
        query = "SELECT * FROM coupang_products WHERE 1=1"
        params = []

        if keyword:
            query += " AND keyword = ?"
            params.append(keyword)

        if seller_type:
            query += " AND seller_type = ?"
            params.append(seller_type)

        # 정렬
        allowed_order = ["timestamp", "price", "rank", "name"]
        if order_by in allowed_order:
            order_dir = "DESC" if order_dir.upper() == "DESC" else "ASC"
            query += f" ORDER BY {order_by} {order_dir}"

        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [
            CoupangProduct(
                id=row["id"],
                keyword=row["keyword"],
                name=row["name"],
                price=row["price"],
                seller_type=row["seller_type"],
                rating=row["rating"],
                review_count=row["review_count"],
                url=row["url"],
                thumbnail=row["thumbnail"],
                rank=row["rank"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    def get_keywords(self) -> list[dict]:
        """수집된 키워드 목록 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT keyword, COUNT(*) as count, MAX(timestamp) as last_collected
                FROM coupang_products
                GROUP BY keyword
                ORDER BY count DESC
            """).fetchall()

        return [
            {
                "keyword": row["keyword"],
                "count": row["count"],
                "last_collected": row["last_collected"],
            }
            for row in rows
        ]

    def get_stats(self) -> dict:
        """통계 조회"""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM coupang_products").fetchone()[0]
            keywords = conn.execute("SELECT COUNT(DISTINCT keyword) FROM coupang_products").fetchone()[0]

            by_seller = conn.execute("""
                SELECT seller_type, COUNT(*) as count
                FROM coupang_products
                GROUP BY seller_type
            """).fetchall()

        return {
            "total_products": total,
            "total_keywords": keywords,
            "by_seller_type": {row[0]: row[1] for row in by_seller},
        }

    def delete_products(self, keyword: str) -> int:
        """특정 키워드의 상품 삭제"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM coupang_products WHERE keyword = ?", (keyword,))
                conn.commit()
                return conn.total_changes

    def delete_all_products(self) -> int:
        """모든 상품 삭제"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM coupang_products")
                conn.commit()
                return conn.total_changes

    def get_product_count(self, keyword: Optional[str] = None) -> int:
        """상품 수 조회"""
        with sqlite3.connect(self.db_path) as conn:
            if keyword:
                count = conn.execute(
                    "SELECT COUNT(*) FROM coupang_products WHERE keyword = ?",
                    (keyword,)
                ).fetchone()[0]
            else:
                count = conn.execute("SELECT COUNT(*) FROM coupang_products").fetchone()[0]
        return count

    def mark_keyword_collected(self, keyword: str, count: int):
        """키워드 수집 완료 표시"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO collected_keywords (keyword, product_count, last_collected, status)
                    VALUES (?, ?, ?, 'completed')
                """, (keyword, count, datetime.utcnow().isoformat()))
                conn.commit()


# 싱글톤 인스턴스
_db_service: Optional[CoupangDBService] = None


def get_coupang_db() -> CoupangDBService:
    """DB 서비스 싱글톤 인스턴스 반환"""
    global _db_service
    if _db_service is None:
        _db_service = CoupangDBService()
    return _db_service
