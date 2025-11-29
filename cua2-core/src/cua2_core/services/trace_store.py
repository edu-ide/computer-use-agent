"""
Trace 저장소 - 노드별 실행 trace 저장 및 재사용

기능:
- 노드별 trace 저장 (JSON 형식)
- 캐시 키 기반 trace 검색
- 성공한 trace만 재사용 가능
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class NodeTrace:
    """노드 실행 trace"""
    workflow_id: str
    node_id: str
    cache_key: str  # 파라미터 기반 캐시 키
    success: bool
    steps: List[Dict[str, Any]] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    used_count: int = 0  # 재사용 횟수
    last_used_at: Optional[str] = None


class TraceStore:
    """
    Trace 저장소

    노드별 실행 trace를 저장하고 재사용 가능하게 관리합니다.
    """

    _instance: Optional["TraceStore"] = None

    def __init__(self, store_dir: str = "/tmp/trace_store"):
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, NodeTrace] = {}
        self._load_cache()

    @classmethod
    def get_instance(cls, store_dir: str = "/tmp/trace_store") -> "TraceStore":
        if cls._instance is None:
            cls._instance = cls(store_dir)
        return cls._instance

    def _load_cache(self):
        """디스크에서 캐시 로드"""
        try:
            index_file = self._store_dir / "index.json"
            if index_file.exists():
                with open(index_file, "r", encoding="utf-8") as f:
                    index = json.load(f)
                    for key, trace_data in index.items():
                        self._cache[key] = NodeTrace(**trace_data)
        except Exception as e:
            print(f"[TraceStore] 캐시 로드 실패: {e}")

    def _save_cache(self):
        """캐시를 디스크에 저장"""
        try:
            index_file = self._store_dir / "index.json"
            index = {key: asdict(trace) for key, trace in self._cache.items()}
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[TraceStore] 캐시 저장 실패: {e}")

    def generate_cache_key(
        self,
        workflow_id: str,
        node_id: str,
        params: Dict[str, Any],
        key_params: List[str] = None,
    ) -> str:
        """
        캐시 키 생성

        Args:
            workflow_id: 워크플로우 ID
            node_id: 노드 ID
            params: 워크플로우 파라미터
            key_params: 캐시 키에 사용할 파라미터 목록 (None이면 전체 사용)

        Returns:
            해시된 캐시 키
        """
        # 키에 사용할 파라미터만 추출
        if key_params:
            filtered_params = {k: params.get(k) for k in key_params if k in params}
        else:
            filtered_params = params

        # 해시 생성
        key_data = f"{workflow_id}:{node_id}:{json.dumps(filtered_params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def save_trace(
        self,
        workflow_id: str,
        node_id: str,
        cache_key: str,
        success: bool,
        steps: List[Dict[str, Any]],
        data: Dict[str, Any] = None,
    ) -> NodeTrace:
        """
        Trace 저장

        Args:
            workflow_id: 워크플로우 ID
            node_id: 노드 ID
            cache_key: 캐시 키
            success: 성공 여부
            steps: VLM 스텝 로그
            data: 노드 결과 데이터

        Returns:
            저장된 NodeTrace
        """
        trace = NodeTrace(
            workflow_id=workflow_id,
            node_id=node_id,
            cache_key=cache_key,
            success=success,
            steps=steps,
            data=data or {},
        )

        # 성공한 trace만 캐시에 저장 (재사용 가능)
        if success:
            self._cache[cache_key] = trace
            self._save_cache()

        # 개별 trace 파일도 저장 (디버깅/히스토리용)
        trace_file = self._store_dir / f"{cache_key}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        try:
            with open(trace_file, "w", encoding="utf-8") as f:
                json.dump(asdict(trace), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[TraceStore] Trace 파일 저장 실패: {e}")

        return trace

    def get_trace(self, cache_key: str) -> Optional[NodeTrace]:
        """
        캐시된 Trace 조회

        Args:
            cache_key: 캐시 키

        Returns:
            캐시된 NodeTrace (없으면 None)
        """
        return self._cache.get(cache_key)

    def get_reusable_trace(
        self,
        workflow_id: str,
        node_id: str,
        params: Dict[str, Any],
        key_params: List[str] = None,
    ) -> Optional[NodeTrace]:
        """
        재사용 가능한 Trace 조회

        Args:
            workflow_id: 워크플로우 ID
            node_id: 노드 ID
            params: 워크플로우 파라미터
            key_params: 캐시 키에 사용할 파라미터 목록

        Returns:
            재사용 가능한 NodeTrace (없으면 None)
        """
        cache_key = self.generate_cache_key(workflow_id, node_id, params, key_params)
        trace = self._cache.get(cache_key)

        if trace and trace.success:
            # 사용 횟수 업데이트
            trace.used_count += 1
            trace.last_used_at = datetime.now().isoformat()
            self._save_cache()
            return trace

        return None

    def list_traces(
        self,
        workflow_id: str = None,
        node_id: str = None,
        success_only: bool = True,
    ) -> List[NodeTrace]:
        """
        Trace 목록 조회

        Args:
            workflow_id: 필터링할 워크플로우 ID
            node_id: 필터링할 노드 ID
            success_only: 성공한 것만 조회

        Returns:
            NodeTrace 목록
        """
        traces = list(self._cache.values())

        if workflow_id:
            traces = [t for t in traces if t.workflow_id == workflow_id]
        if node_id:
            traces = [t for t in traces if t.node_id == node_id]
        if success_only:
            traces = [t for t in traces if t.success]

        return traces

    def delete_trace(self, cache_key: str) -> bool:
        """
        Trace 삭제

        Args:
            cache_key: 삭제할 캐시 키

        Returns:
            삭제 성공 여부
        """
        if cache_key in self._cache:
            del self._cache[cache_key]
            self._save_cache()
            return True
        return False

    def clear_all(self):
        """모든 캐시 삭제"""
        self._cache.clear()
        self._save_cache()


# 싱글톤 인스턴스 가져오기
def get_trace_store(store_dir: str = "/tmp/trace_store") -> TraceStore:
    return TraceStore.get_instance(store_dir)
