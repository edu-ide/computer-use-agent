"""
Library Service
---------------
Provides information about available Standard Nodes and Agent Types.
"""

from typing import List, Dict, Any

class LibraryService:
    @staticmethod
    def get_standard_nodes() -> List[Dict[str, Any]]:
        """Returns a list of available standard node templates."""
        return [
            {
                "type": "navigation",
                "name": "Navigation Node",
                "display_name_ko": "페이지 이동 (Navigation)",
                "description": "Navigate to a specific URL and verify it loads.",
                "description_ko": "특정 URL로 이동하고 페이지 로딩을 검증합니다.",
                "factory_function": "create_navigation_node",
                "parameters": [
                    {"name": "name", "type": "str", "description": "Node name (unique id)", "description_ko": "노드 이름 (고유 ID)"},
                    {"name": "url", "type": "str", "description": "Target URL", "description_ko": "이동할 대상 URL"},
                    {"name": "display_name", "type": "str", "default": "Navigate", "description": "Display name in UI", "description_ko": "UI에 표시될 이름"},
                ]
            },
            {
                "type": "search",
                "name": "Search Node",
                "display_name_ko": "검색 (Search)",
                "description": "Input a keyword into a search bar and execute the search.",
                "description_ko": "검색창에 키워드를 입력하고 검색을 실행합니다.",
                "factory_function": "create_search_node",
                "parameters": [
                    {"name": "name", "type": "str", "description": "Node name (unique id)", "description_ko": "노드 이름 (고유 ID)"},
                    {"name": "keyword_param", "type": "str", "description": "Parameter key for the keyword", "description_ko": "키워드 파라미터 키 (예: 'keyword')"},
                    {"name": "display_name", "type": "str", "default": "Search", "description": "Display name in UI", "description_ko": "UI에 표시될 이름"},
                ]
            },
            {
                "type": "analysis",
                "name": "Analysis Node",
                "display_name_ko": "페이지 분석 (Analysis)",
                "description": "Analyze a page, extract data (preferring JS), and optionally perform actions.",
                "description_ko": "페이지를 분석하여 데이터를 추출하거나(JS 권장) 작업을 수행합니다.",
                "factory_function": "create_analysis_node",
                "parameters": [
                    {"name": "name", "type": "str", "description": "Node name (unique id)", "description_ko": "노드 이름 (고유 ID)"},
                    {"name": "goal", "type": "str", "description": "High-level goal of the analysis", "description_ko": "분석의 상위 목표"},
                    {"name": "extraction_details", "type": "str", "description": "Specifics on what to extract", "description_ko": "추출할 데이터 상세 내용"},
                    {"name": "output_details", "type": "str", "description": "JSON structure description", "description_ko": "반환할 JSON 구조 설명"},
                    {"name": "pagination_details", "type": "str", "description": "Instructions for pagination", "description_ko": "페이지 이동(Pagination) 지침 (선택 사항)"},
                ]
            }
        ]

    @staticmethod
    def get_agent_types() -> List[Dict[str, Any]]:
        """Returns a list of available agent types."""
        return [
            {
                "type": "VLMAgent",
                "name": "VLM Agent",
                "name_ko": "VLM 에이전트",
                "description": "Vision-Language Model based agent. Uses screenshots to understand the page and perform actions.",
                "description_ko": "시각-언어 모델(VLM) 기반 에이전트입니다. 스크린샷을 통해 페이지를 이해하고 작업을 수행합니다.",
                "capabilities": ["Vision Analysis", "Click", "Type", "Scroll", "JS Execution"],
                "capabilities_ko": ["시각 분석", "클릭", "타이핑", "스크롤", "JS 실행"],
                "recommended_for": ["General Navigation", "Complex UI Interaction", "Visual Verification"],
                "recommended_for_ko": ["일반적인 웹 탐색", "복잡한 UI 상호작용", "시각적 검증"]
            },
            {
                "type": "SearchAgent",
                "name": "Search Agent",
                "name_ko": "검색 에이전트",
                "description": "Specialized agent for search operations (Future implementation).",
                "description_ko": "검색 작업에 특화된 에이전트입니다 (구현 예정).",
                "capabilities": ["Search Query Optimization", "Result Parsing"],
                "capabilities_ko": ["검색어 최적화", "결과 파싱"],
                "recommended_for": ["Search Tasks"],
                "recommended_for_ko": ["검색 중심 작업"]
            },
            {
                "type": "AnalysisAgent",
                "name": "Analysis Agent",
                "name_ko": "분석 에이전트",
                "description": "Specialized agent for deep page analysis and data extraction (Future implementation).",
                "description_ko": "심층 페이지 분석 및 데이터 추출에 특화된 에이전트입니다 (구현 예정).",
                "capabilities": ["DOM Parsing", "Data Extraction", "Structure Analysis"],
                "capabilities_ko": ["DOM 파싱", "데이터 추출", "구조 분석"],
                "recommended_for": ["Data Scraping", "Content Analysis"],
                "recommended_for_ko": ["데이터 수집", "콘텐츠 분석"]
            }
        ]
