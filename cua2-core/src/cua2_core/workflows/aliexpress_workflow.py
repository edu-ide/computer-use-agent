import asyncio
import os
import subprocess
from typing import Dict, List, Any
from .workflow_base import (
    WorkflowBase,
    WorkflowConfig,
    WorkflowNode,
    WorkflowState,
    NodeResult,
    VLMErrorType,
)

class AliExpressWorkflow(WorkflowBase):
    def __init__(self, agent_runner=None):
        super().__init__()
        self._agent_runner = agent_runner

    @property
    def config(self) -> WorkflowConfig:
        return WorkflowConfig(
            id="aliexpress-video",
            name="AliExpress Video Scraper",
            description="Scrape product videos from AliExpress using the existing script.",
            icon="Video",
            color="#FF4747",
            category="automation",
            parameters=[
                {
                    "name": "keyword",
                    "type": "string",
                    "label": "Search Keyword or Batch File",
                    "placeholder": "e.g. vacuum cleaner or /path/to/file.xlsx",
                    "default": "/home/sk/ws/mcp-playwright/15.aliexpress_video/keywords_to_search.xlsx",
                    "required": True,
                },
                {
                    "name": "pages",
                    "type": "number",
                    "label": "Pages to Scrape",
                    "default": 1,
                    "min": 1,
                    "max": 10,
                }
            ],
        )

    @property
    def nodes(self) -> List[WorkflowNode]:
        return [
            WorkflowNode(
                name="run_scraper",
                display_name="Run Scraper Script",
                description="Executes the aliexpress_video.py script",
                node_type="process",
                timeout_sec=600,
                avg_duration_sec=60,
            )
        ]

    @property
    def start_node(self) -> str:
        return "run_scraper"

    async def execute_node(self, node_name: str, state: WorkflowState) -> NodeResult:
        if node_name == "run_scraper":
            params = state.get("parameters", {})
            keyword = params.get("keyword", "")
            pages = params.get("pages", 1)
            
            script_path = "/home/sk/ws/mcp-playwright/15.aliexpress_video/aliexpress_video.py"
            
            # Check if keyword is actually a batch file path
            is_batch_file = False
            if keyword and (keyword.endswith('.xlsx') or keyword.endswith('.xls')) and os.path.exists(keyword):
                is_batch_file = True
                
            if is_batch_file:
                cmd = [
                    "python", 
                    script_path,
                    "--batch-file", keyword,
                    "--pages", str(pages)
                ]
            else:
                cmd = [
                    "python", 
                    script_path,
                    keyword,
                    "--pages", str(pages)
                ]

            print(f"[AliExpressWorkflow] Running command: {' '.join(cmd)}")
            
            try:
                # Run subprocess
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    output = stdout.decode()
                    print(output)
                    return NodeResult(success=True, data={"output": output})
                else:
                    error = stderr.decode()
                    print(f"[AliExpressWorkflow] Error: {error}")
                    return NodeResult(success=False, error=error)
                    
            except Exception as e:
                return NodeResult(success=False, error=str(e))

        return NodeResult(success=False, error=f"Unknown node: {node_name}")
