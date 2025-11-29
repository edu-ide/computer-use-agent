"""
Multi-Agent System

smolagents 기반 다중 에이전트 시스템:
- 전문화된 에이전트들 (SearchAgent, AnalysisAgent, ValidationAgent)
- Manager Agent가 전체 조정
- 복잡한 작업을 분담하여 병렬/순차 처리

Reference: smolagents multi-agent orchestration
"""

from .base_agent import BaseSpecializedAgent, AgentResult
from .search_agent import SearchAgent
from .analysis_agent import AnalysisAgent
from .validation_agent import ValidationAgent
from .manager_agent import ManagerAgent, WorkflowStep, WorkflowResult

__all__ = [
    "BaseSpecializedAgent",
    "AgentResult",
    "SearchAgent",
    "AnalysisAgent",
    "ValidationAgent",
    "ManagerAgent",
    "WorkflowStep",
    "WorkflowResult",
]
