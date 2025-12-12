"""
Coupang API Workflow - Wrapping existing scraper logic
"""

import sys
import os
import asyncio
from typing import Dict, Any, List, Optional
import logging

from .workflow_base import (
    WorkflowBase,
    WorkflowConfig,
    WorkflowNode,
    WorkflowState,
    NodeResult,
    VLMErrorType,
)

# Add the external project path to access scraper and config modules
EXTERNAL_PROJECT_PATH = "/home/sk/ws/mcp-playwright/11_coupang_wing_web"
if EXTERNAL_PROJECT_PATH not in sys.path:
    # Append to beginning to ensure priority or end? 
    # Appending to end is safer to avoid masking core modules
    sys.path.append(EXTERNAL_PROJECT_PATH)

# Try importing, handle failure gracefully
try:
    from scraper import CoupangScraper
    SCRAPER_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import CoupangScraper: {e}")
    SCRAPER_AVAILABLE = False


class MockSocketIO:
    """Mock SocketIO for capturing scraper logs and events"""
    def __init__(self):
        self.logs = []
        self.results = []
        self.status = "initialized"
        self._callbacks = {}

    def on(self, event_name, callback):
        self._callbacks[event_name] = callback

    def emit(self, event, data):
        """Capture events from scraper"""
        if event == 'log':
            log_entry = f"[{data.get('level', 'INFO').upper()}] {data.get('message', '')}"
            self.logs.append(log_entry)
            print(f"[CoupangScraper] {log_entry}")
            
        elif event == 'result_update':
            count = data.get('count', 0)
            self.status = f"Collecting... ({count} items)"
            
        elif event == 'scraping_complete':
            self.status = "completed"
            if isinstance(data, dict):
                # Data structure from scraper:
                # {
                #     'success': True,
                #     'count': len(scraping_results),
                #     'results': scraping_results,
                #     'similar_items': similar_items,
                #     ...
                # }
                self.results = data


class CoupangApiWorkflow(WorkflowBase):
    """
    Coupang Product Collection Workflow (Native API Version)
    Wraps the existing 11_coupang_wing_web logic.
    """

    def __init__(self, agent_runner=None):
        super().__init__()
        self._agent_runner = agent_runner
        self._mock_socket = MockSocketIO()

    @property
    def config(self) -> WorkflowConfig:
        return WorkflowConfig(
            id="coupang-api-collect",
            name="쿠팡 상품 수집 (API)",
            description="기존 스크래퍼를 사용하여 쿠팡 상품을 수집합니다.",
            icon="Server",
            color="#00C73C",
            category="automation",
            parameters=[
                {
                    "name": "query",
                    "type": "string",
                    "label": "검색어",
                    "placeholder": "예: 노트북",
                    "required": True,
                },
                {
                    "name": "max_results",
                    "type": "number",
                    "label": "최대 수집 개수",
                    "default": 20,
                    "min": 5,
                    "max": 100,
                },
                {
                    "name": "headless",
                    "type": "boolean",
                    "label": "헤드리스 모드 (창 숨기기)",
                    "default": False,
                },
                {
                    "name": "use_existing_browser",
                    "type": "boolean",
                    "label": "기존 브라우저 사용",
                    "default": False,
                }
            ],
        )

    @property
    def nodes(self) -> List[WorkflowNode]:
        return [
            WorkflowNode(
                name="run_scraper",
                display_name="스크래퍼 실행",
                description="쿠팡 스크래퍼 실행 중...",
                node_type="process",
                on_success="complete",
                on_failure="error_handler",
                timeout_sec=600,  # 10분
                avg_duration_sec=60,
            ),
            WorkflowNode(
                name="complete",
                display_name="완료",
                description="수집 완료",
                node_type="end",
            ),
            WorkflowNode(
                name="error_handler",
                display_name="에러",
                description="수집 실패",
                node_type="error",
            ),
        ]

    @property
    def start_node(self) -> str:
        return "run_scraper"

    async def execute_node(self, node_name: str, state: WorkflowState) -> NodeResult:
        """Execute workflow node"""
        if node_name == "run_scraper":
            return await self._run_scraper_node(state)
        
        elif node_name == "complete":
            return NodeResult(success=True, data={"status": "completed"})
            
        elif node_name == "error_handler":
            return NodeResult(success=False, error="Workflow failed")

        return NodeResult(success=False, error=f"Unknown node: {node_name}")

    async def _run_scraper_node(self, state: WorkflowState) -> NodeResult:
        """Run the scraper logic"""
        if not SCRAPER_AVAILABLE:
            return NodeResult(
                success=False, 
                error="CoupangScraper module not found. Check path: " + EXTERNAL_PROJECT_PATH
            )
            
        params = state.get("parameters", {})
        query = params.get("query", "test")
        
        # Prepare search params as expected by CoupangScraper
        search_params = {
            'query': query,
            'max_results': params.get("max_results", 20),
            'headless': params.get("headless", False),
            'use_existing_browser': params.get("use_existing_browser", False),
            'cdp_url': 'http://localhost:9222', # Default
            # Add other defaults if needed
            'sort': 'best',
            'min_price': 0,
            'max_price': 999999999,
        }
        
        print(f"[CoupangApiWorkflow] Starting scraper with: {search_params}")
        
        try:
            # Instantiate scraper with mock socket
            scraper = CoupangScraper(self._mock_socket)
            
            # Run scraper (it's synchronous but handles its own event loop internally?
            # Wait, scraper.run() creates a new event loop: loop = asyncio.new_event_loop()
            # If we are already in an async loop (executed by uvicorn), this might cause issues 
            # if scraper.run tries to use asyncio.new_event_loop() which might not nest well.
            # However, scraper.run() calls loop.run_until_complete(self.scrape(search_params))
            # If we call this from an async function, we are blocking the main loop.
            # We should probably run this in a thread executor to avoid blocking the main server loop.
            
            def run_wrapper():
                return scraper.run(search_params)
            
            # thread_pool_executor execution
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, run_wrapper)
            
            # Result from scraper.run() is directly the results list or dict
            # Scraper.run returns: results (list) or dict?
            # Looking at code: return results (line 886)
            # And results comes from self.scrape which returns:
            # {'main_results': results, 'similar_items': ..., ...} (line 796)
            
            output_data = {}
            if isinstance(result, dict):
                output_data = result
                item_count = len(result.get('main_results', []))
            else:
                # Fallback if list
                output_data = {'results': result}
                item_count = len(result)
            
            return NodeResult(
                success=True,
                data={
                    "count": item_count,
                    "results_summary": f"Collected {item_count} items for '{query}'",
                    "full_data": output_data,
                    "logs": self._mock_socket.logs
                }
            )
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[CoupangApiWorkflow] Error: {e}\n{tb}")
            return NodeResult(
                success=False,
                error=f"Scraper failed: {str(e)}",
                data={"logs": self._mock_socket.logs}
            )
