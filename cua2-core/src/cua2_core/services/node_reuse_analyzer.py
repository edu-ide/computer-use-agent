"""
노드 재사용 분석기 - VLM이 노드 실행 결과를 분석하여 재사용 설정을 자동으로 학습

ToolOrchestra 논문의 아이디어를 적용:
- 작은 모델(orchestrator)이 실행 결과를 분석
- 재사용 가능 여부, 메모리 공유 필요성 등을 자동 판단
- 학습된 설정을 Letta 메모리에 저장하여 다음 실행에 활용
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ReuseDecision(Enum):
    """재사용 결정 유형"""
    REUSABLE = "reusable"  # 결과가 일관적, 재사용 가능
    NOT_REUSABLE = "not_reusable"  # 외부 상태 의존, 재사용 불가
    NEEDS_MEMORY = "needs_memory"  # 이전 컨텍스트 필요
    UNCERTAIN = "uncertain"  # 판단 불가, 더 많은 데이터 필요


@dataclass
class NodeExecutionAnalysis:
    """노드 실행 분석 결과"""
    node_id: str
    workflow_id: str

    # 실행 특성 분석
    uses_external_state: bool = False  # 외부 상태(웹페이지 등) 의존
    uses_parameters: bool = False  # 파라미터 사용
    parameter_keys: List[str] = field(default_factory=list)  # 사용된 파라미터
    produces_data: bool = False  # 데이터 생성 여부
    requires_previous_context: bool = False  # 이전 컨텍스트 필요

    # 실행 결과 분석
    success_rate: float = 0.0  # 성공률 (0~1)
    consistency_score: float = 0.0  # 결과 일관성 (0~1)
    avg_steps: float = 0.0  # 평균 스텝 수

    # 결정
    decision: ReuseDecision = ReuseDecision.UNCERTAIN
    confidence: float = 0.0  # 확신도 (0~1)
    reason: str = ""

    # 추천 설정
    recommended_reusable: bool = False
    recommended_reuse_trace: bool = False
    recommended_share_memory: bool = False
    recommended_cache_key_params: List[str] = field(default_factory=list)


@dataclass
class NodeExecutionRecord:
    """노드 실행 기록"""
    node_id: str
    workflow_id: str
    execution_id: str
    success: bool
    steps_count: int
    parameters: Dict[str, Any]
    data_produced: Dict[str, Any]
    error: Optional[str] = None


class NodeReuseAnalyzer:
    """
    노드 재사용 분석기

    VLM 또는 규칙 기반으로 노드 실행 결과를 분석하여
    재사용 설정을 자동으로 학습합니다.
    """

    # 분석에 필요한 최소 실행 횟수
    MIN_EXECUTIONS_FOR_ANALYSIS = 3

    # 재사용 결정 임계값
    SUCCESS_RATE_THRESHOLD = 0.8  # 80% 이상 성공
    CONSISTENCY_THRESHOLD = 0.7  # 70% 이상 일관성

    def __init__(self, use_vlm: bool = False, vlm_client=None):
        """
        Args:
            use_vlm: VLM을 사용한 분석 여부 (False면 규칙 기반)
            vlm_client: VLM 클라이언트 (use_vlm=True일 때 필요)
        """
        self._use_vlm = use_vlm
        self._vlm_client = vlm_client

        # 노드별 실행 기록 저장
        self._execution_history: Dict[str, List[NodeExecutionRecord]] = {}

        # 노드별 분석 결과 캐시
        self._analysis_cache: Dict[str, NodeExecutionAnalysis] = {}

    def record_execution(
        self,
        node_id: str,
        workflow_id: str,
        execution_id: str,
        success: bool,
        steps_count: int,
        parameters: Dict[str, Any],
        data_produced: Dict[str, Any],
        error: Optional[str] = None,
    ):
        """노드 실행 기록 저장"""
        key = f"{workflow_id}:{node_id}"

        if key not in self._execution_history:
            self._execution_history[key] = []

        record = NodeExecutionRecord(
            node_id=node_id,
            workflow_id=workflow_id,
            execution_id=execution_id,
            success=success,
            steps_count=steps_count,
            parameters=parameters,
            data_produced=data_produced,
            error=error,
        )

        self._execution_history[key].append(record)

        # 분석 캐시 무효화
        if key in self._analysis_cache:
            del self._analysis_cache[key]

        logger.debug(f"노드 실행 기록: {key}, success={success}")

    def analyze_node(
        self,
        node_id: str,
        workflow_id: str,
        node_instruction: Optional[str] = None,
    ) -> NodeExecutionAnalysis:
        """
        노드 실행 기록을 분석하여 재사용 설정 추천

        Args:
            node_id: 노드 ID
            workflow_id: 워크플로우 ID
            node_instruction: 노드 instruction (VLM 분석용)

        Returns:
            NodeExecutionAnalysis: 분석 결과 및 추천 설정
        """
        key = f"{workflow_id}:{node_id}"

        # 캐시 확인
        if key in self._analysis_cache:
            return self._analysis_cache[key]

        # 실행 기록 가져오기
        history = self._execution_history.get(key, [])

        if len(history) < self.MIN_EXECUTIONS_FOR_ANALYSIS:
            # 데이터 부족
            return NodeExecutionAnalysis(
                node_id=node_id,
                workflow_id=workflow_id,
                decision=ReuseDecision.UNCERTAIN,
                confidence=0.0,
                reason=f"데이터 부족 ({len(history)}/{self.MIN_EXECUTIONS_FOR_ANALYSIS})",
            )

        # 분석 수행
        if self._use_vlm and self._vlm_client:
            analysis = self._analyze_with_vlm(history, node_instruction)
        else:
            analysis = self._analyze_with_rules(history)

        # 캐시 저장
        self._analysis_cache[key] = analysis

        return analysis

    def _analyze_with_rules(
        self,
        history: List[NodeExecutionRecord],
    ) -> NodeExecutionAnalysis:
        """규칙 기반 분석"""
        if not history:
            return NodeExecutionAnalysis(
                node_id="",
                workflow_id="",
                decision=ReuseDecision.UNCERTAIN,
            )

        first = history[0]
        node_id = first.node_id
        workflow_id = first.workflow_id

        # 기본 통계 계산
        success_count = sum(1 for r in history if r.success)
        success_rate = success_count / len(history)

        avg_steps = sum(r.steps_count for r in history) / len(history)

        # 파라미터 분석
        all_params = set()
        for r in history:
            all_params.update(r.parameters.keys())

        # 결과 일관성 분석 (같은 파라미터로 같은 결과?)
        consistency_score = self._calculate_consistency(history)

        # 이전 컨텍스트 필요 여부 판단
        # - 첫 번째 노드가 아니고
        # - 파라미터가 적거나 없으면 이전 컨텍스트에 의존할 가능성
        requires_context = len(all_params) == 0 and node_id != "open_coupang"

        # 외부 상태 의존 여부 판단
        # - 성공률이 낮거나 결과가 불일관하면 외부 상태 의존
        uses_external = success_rate < 0.9 or consistency_score < 0.5

        # 결정
        decision = ReuseDecision.UNCERTAIN
        confidence = 0.0
        reason = ""

        # 추천 설정
        recommended_reusable = False
        recommended_reuse_trace = False
        recommended_share_memory = requires_context
        recommended_cache_params = list(all_params)

        if success_rate >= self.SUCCESS_RATE_THRESHOLD:
            if consistency_score >= self.CONSISTENCY_THRESHOLD:
                decision = ReuseDecision.REUSABLE
                confidence = min(success_rate, consistency_score)
                reason = f"높은 성공률({success_rate:.1%})과 일관성({consistency_score:.1%})"
                recommended_reusable = True
                recommended_reuse_trace = True
            elif requires_context:
                decision = ReuseDecision.NEEDS_MEMORY
                confidence = success_rate * 0.8
                reason = f"성공률 높지만({success_rate:.1%}) 이전 컨텍스트 필요"
                recommended_reusable = False
                recommended_share_memory = True
            else:
                decision = ReuseDecision.NOT_REUSABLE
                confidence = 0.5
                reason = f"결과 불일관({consistency_score:.1%}), 외부 상태 의존 가능"
        else:
            decision = ReuseDecision.NOT_REUSABLE
            confidence = 1 - success_rate
            reason = f"낮은 성공률({success_rate:.1%})"

        return NodeExecutionAnalysis(
            node_id=node_id,
            workflow_id=workflow_id,
            uses_external_state=uses_external,
            uses_parameters=len(all_params) > 0,
            parameter_keys=list(all_params),
            produces_data=any(r.data_produced for r in history),
            requires_previous_context=requires_context,
            success_rate=success_rate,
            consistency_score=consistency_score,
            avg_steps=avg_steps,
            decision=decision,
            confidence=confidence,
            reason=reason,
            recommended_reusable=recommended_reusable,
            recommended_reuse_trace=recommended_reuse_trace,
            recommended_share_memory=recommended_share_memory,
            recommended_cache_key_params=recommended_cache_params,
        )

    def _calculate_consistency(
        self,
        history: List[NodeExecutionRecord],
    ) -> float:
        """결과 일관성 계산"""
        if len(history) < 2:
            return 1.0

        # 같은 파라미터로 실행한 결과들을 그룹화
        param_groups: Dict[str, List[bool]] = {}

        for record in history:
            # 파라미터를 키로 변환
            param_key = json.dumps(record.parameters, sort_keys=True)

            if param_key not in param_groups:
                param_groups[param_key] = []

            param_groups[param_key].append(record.success)

        # 각 그룹 내 일관성 계산
        consistencies = []
        for results in param_groups.values():
            if len(results) > 1:
                # 모두 같은 결과면 1.0, 아니면 다수결 비율
                most_common = max(set(results), key=results.count)
                consistency = results.count(most_common) / len(results)
                consistencies.append(consistency)

        if not consistencies:
            return 1.0

        return sum(consistencies) / len(consistencies)

    async def _analyze_with_vlm(
        self,
        history: List[NodeExecutionRecord],
        node_instruction: Optional[str] = None,
    ) -> NodeExecutionAnalysis:
        """VLM을 사용한 분석 (ToolOrchestra 스타일)"""
        if not history:
            return NodeExecutionAnalysis(
                node_id="",
                workflow_id="",
                decision=ReuseDecision.UNCERTAIN,
            )

        first = history[0]

        # VLM에 분석 요청
        prompt = self._build_analysis_prompt(history, node_instruction)

        try:
            # VLM 호출 (실제 구현 필요)
            if self._vlm_client:
                response = await self._vlm_client.analyze(prompt)
                return self._parse_vlm_response(response, first.node_id, first.workflow_id)
        except Exception as e:
            logger.error(f"VLM 분석 실패: {e}")

        # VLM 실패 시 규칙 기반으로 폴백
        return self._analyze_with_rules(history)

    def _build_analysis_prompt(
        self,
        history: List[NodeExecutionRecord],
        node_instruction: Optional[str] = None,
    ) -> str:
        """VLM 분석용 프롬프트 생성"""
        first = history[0]

        # 실행 기록 요약
        summary = {
            "total_executions": len(history),
            "success_count": sum(1 for r in history if r.success),
            "avg_steps": sum(r.steps_count for r in history) / len(history),
            "unique_params": len(set(
                json.dumps(r.parameters, sort_keys=True) for r in history
            )),
            "errors": [r.error for r in history if r.error],
        }

        prompt = f"""당신은 워크플로우 최적화 전문가입니다.
다음 노드의 실행 기록을 분석하여 재사용 설정을 추천해주세요.

## 노드 정보
- 노드 ID: {first.node_id}
- 워크플로우: {first.workflow_id}
{f'- Instruction: {node_instruction}' if node_instruction else ''}

## 실행 기록 요약
{json.dumps(summary, indent=2, ensure_ascii=False)}

## 분석 기준
1. **reusable**: 노드 결과가 일관적이고 외부 상태에 의존하지 않을 때
2. **reuse_trace**: 동일 입력에 동일 출력이 보장될 때 (캐시 사용)
3. **share_memory**: 이전 노드의 컨텍스트가 필요할 때
4. **cache_key_params**: 결과에 영향을 주는 파라미터 목록

## 응답 형식 (JSON)
{{
    "decision": "reusable|not_reusable|needs_memory|uncertain",
    "confidence": 0.0~1.0,
    "reason": "판단 이유",
    "recommended_reusable": true/false,
    "recommended_reuse_trace": true/false,
    "recommended_share_memory": true/false,
    "recommended_cache_key_params": ["param1", "param2"]
}}
"""
        return prompt

    def _parse_vlm_response(
        self,
        response: str,
        node_id: str,
        workflow_id: str,
    ) -> NodeExecutionAnalysis:
        """VLM 응답 파싱"""
        try:
            # JSON 추출
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                decision_map = {
                    "reusable": ReuseDecision.REUSABLE,
                    "not_reusable": ReuseDecision.NOT_REUSABLE,
                    "needs_memory": ReuseDecision.NEEDS_MEMORY,
                    "uncertain": ReuseDecision.UNCERTAIN,
                }

                return NodeExecutionAnalysis(
                    node_id=node_id,
                    workflow_id=workflow_id,
                    decision=decision_map.get(
                        data.get("decision", "uncertain"),
                        ReuseDecision.UNCERTAIN
                    ),
                    confidence=float(data.get("confidence", 0.5)),
                    reason=data.get("reason", ""),
                    recommended_reusable=data.get("recommended_reusable", False),
                    recommended_reuse_trace=data.get("recommended_reuse_trace", False),
                    recommended_share_memory=data.get("recommended_share_memory", False),
                    recommended_cache_key_params=data.get("recommended_cache_key_params", []),
                )
        except Exception as e:
            logger.error(f"VLM 응답 파싱 실패: {e}")

        return NodeExecutionAnalysis(
            node_id=node_id,
            workflow_id=workflow_id,
            decision=ReuseDecision.UNCERTAIN,
        )

    def get_recommended_settings(
        self,
        node_id: str,
        workflow_id: str,
    ) -> Dict[str, Any]:
        """노드의 추천 재사용 설정 반환"""
        analysis = self.analyze_node(node_id, workflow_id)

        return {
            "reusable": analysis.recommended_reusable,
            "reuse_trace": analysis.recommended_reuse_trace,
            "share_memory": analysis.recommended_share_memory,
            "cache_key_params": analysis.recommended_cache_key_params,
            "confidence": analysis.confidence,
            "reason": analysis.reason,
        }

    def to_letta_memory_format(
        self,
        workflow_id: str,
    ) -> str:
        """Letta 메모리에 저장할 형식으로 변환"""
        analyses = []

        for key, analysis in self._analysis_cache.items():
            if analysis.workflow_id == workflow_id:
                analyses.append({
                    "node_id": analysis.node_id,
                    "decision": analysis.decision.value,
                    "confidence": analysis.confidence,
                    "reason": analysis.reason,
                    "settings": {
                        "reusable": analysis.recommended_reusable,
                        "reuse_trace": analysis.recommended_reuse_trace,
                        "share_memory": analysis.recommended_share_memory,
                        "cache_key_params": analysis.recommended_cache_key_params,
                    },
                    "stats": {
                        "success_rate": analysis.success_rate,
                        "consistency": analysis.consistency_score,
                        "avg_steps": analysis.avg_steps,
                    },
                })

        if not analyses:
            return """## 재사용 학습 데이터
아직 학습된 데이터가 없습니다.

### 학습 기준:
- reusable: 노드 결과가 일관적이고 외부 상태에 의존하지 않을 때
- reuse_trace: 동일 입력에 동일 출력이 보장될 때
- share_memory: 이전 노드의 컨텍스트가 필요할 때
- cache_key_params: 결과에 영향을 주는 파라미터 목록
"""

        lines = ["## 재사용 학습 데이터\n"]

        for a in analyses:
            lines.append(f"### {a['node_id']}")
            lines.append(f"- 결정: {a['decision']} (확신도: {a['confidence']:.1%})")
            lines.append(f"- 이유: {a['reason']}")
            lines.append(f"- 추천 설정: reusable={a['settings']['reusable']}, "
                        f"reuse_trace={a['settings']['reuse_trace']}, "
                        f"share_memory={a['settings']['share_memory']}")
            lines.append(f"- 통계: 성공률={a['stats']['success_rate']:.1%}, "
                        f"일관성={a['stats']['consistency']:.1%}\n")

        return "\n".join(lines)


# 싱글톤 인스턴스
_analyzer: Optional[NodeReuseAnalyzer] = None


def get_node_reuse_analyzer(use_vlm: bool = False) -> NodeReuseAnalyzer:
    """노드 재사용 분석기 싱글톤 반환"""
    global _analyzer
    if _analyzer is None:
        _analyzer = NodeReuseAnalyzer(use_vlm=use_vlm)
    return _analyzer
