"""
Standardized Workflow Nodes
---------------------------
This module provides factory functions for creating standardized, reusable workflow nodes.
These nodes are designed to be generic and adaptable to various web automation tasks.

Node Types:
- Navigation: Navigate to a specific URL.
- Search: Perform a search operation.
- Analysis: Analyze a page and extract data (encourages JS execution).
"""

from typing import Optional, List, Any, Dict
from .workflow_base import WorkflowNode

def create_navigation_node(
    name: str,
    url: str,
    display_name: str = "Navigate",
    description: str = "Navigate to URL",
    on_success: Optional[str] = None,
    on_failure: Optional[str] = "error_handler",
    agent_type: str = "VLMAgent",
    model_id: str = "local-qwen3-vl",
    timeout_sec: int = 60,
    avg_duration_sec: int = 10,
) -> WorkflowNode:
    """
    Creates a standardized navigation node.
    
    Goal: Navigate to the specified URL and verify it loads.
    """
    instruction = f"""Goal: Navigate to '{url}'.

1. Use `open_url("{url}")` to open the page.
2. Wait for the page to load.
3. Verify that the page has loaded correctly (e.g., check title or key elements).

IMPORTANT:
- If the page fails to load, report the error.
- Do not proceed until navigation is confirmed."""

    return WorkflowNode(
        name=name,
        display_name=display_name,
        description=description,
        on_success=on_success,
        on_failure=on_failure,
        node_type="vlm",
        agent_type=agent_type,
        model_id=model_id,
        timeout_sec=timeout_sec,
        avg_duration_sec=avg_duration_sec,
        instruction=instruction,
        metadata={
            "url": url,
        },
    )

def create_search_node(
    name: str,
    keyword_param: str,
    display_name: str = "Search",
    description: str = "Search for keyword",
    on_success: Optional[str] = None,
    on_failure: Optional[str] = "error_handler",
    agent_type: str = "VLMAgent",
    model_id: str = "local-qwen3-vl",
    timeout_sec: int = 90,
    avg_duration_sec: int = 30,
) -> WorkflowNode:
    """
    Creates a standardized search node.
    
    Goal: Input a keyword into a search bar and execute the search.
    """
    instruction = f"""Goal: Search for the keyword "{{{keyword_param}}}" on the current site.

1. Locate the search input field.
2. Input the keyword "{{{keyword_param}}}" using `write()`.
3. Execute the search (usually by pressing Enter or clicking a search button).
4. Verify that the search results page is loaded.

IMPORTANT:
- If the search bar is not visible, look for a search icon to expand it.
- Verify the search result page contains results relevant to "{{{keyword_param}}}".
- Do not proceed until the search results are confirmed visible."""

    return WorkflowNode(
        name=name,
        display_name=display_name,
        description=description,
        on_success=on_success,
        on_failure=on_failure,
        node_type="vlm",
        agent_type=agent_type,
        model_id=model_id,
        timeout_sec=timeout_sec,
        avg_duration_sec=avg_duration_sec,
        instruction=instruction,
        metadata={
            "keyword_param": keyword_param,
        },
    )

def create_analysis_node(
    name: str,
    goal: str,
    extraction_details: str,
    output_details: str,
    pagination_details: Optional[str] = None,
    display_name: str = "Analyze",
    description: str = "Analyze page and extract data",
    on_success: Optional[str] = None,
    on_failure: Optional[str] = "error_handler",
    agent_type: str = "VLMAgent",
    model_id: str = "local-qwen3-vl",
    node_type: str = "vlm",  # 'vlm' or 'extract_data'
    clickable: bool = False,  # 클릭하여 상세 보기 가능 여부
    timeout_sec: int = 120,
    avg_duration_sec: int = 60,
) -> WorkflowNode:
    """
    Creates a standardized analysis node with structured configuration.
    
    Args:
        name: Node name
        goal: High-level goal of the analysis
        extraction_details: Specifics on what to extract (e.g., "Extract Name, Price...")
        output_details: JSON structure description
        pagination_details: Instructions for pagination (optional)
        ...
    """
    
    pagination_section = ""
    if pagination_details:
        pagination_section = f"""
2. PAGINATION:
   {pagination_details}"""

    instruction = f"""Goal: {goal}

1. DATA EXTRACTION & ANALYSIS:
   - **PRIMARY:** Use `run_javascript(script)` to extract data or analyze the DOM. It is faster and more accurate.
   - SECONDARY: Use visual analysis if JS execution is not applicable.
   - **TARGET:** {extraction_details}

{pagination_section}

3. OUTPUT:
   - Return a JSON object with:
     {output_details}

IMPORTANT:
- Verify the page state before and after actions.
- If data extraction fails, check if the page is fully loaded.
- Prefer `run_javascript` over visual analysis for accuracy and speed."""

    return WorkflowNode(
        name=name,
        display_name=display_name,
        description=description,
        on_success=on_success,
        on_failure=on_failure,
        node_type=node_type,  # Use parameter instead of hardcoded "vlm"
        agent_type=agent_type,
        model_id=model_id,
        clickable=clickable,  # 클릭 가능 여부
        timeout_sec=timeout_sec,
        avg_duration_sec=avg_duration_sec,
        instruction=instruction,
        metadata={
            "goal": goal,
            "extraction_details": extraction_details,
            "pagination_details": pagination_details,
            "output_details": output_details,
        },
    )
