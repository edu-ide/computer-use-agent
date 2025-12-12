import asyncio
import os
import logging
from cua2_core.workflows.google_search_workflow import GoogleSearchWorkflow
from cua2_core.services.agent_activity_log import ActivityType

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_workflow():
    print("-" * 50)
    print("Google Search Workflow Verification (VLMAgentRunner + GELab)")
    print("-" * 50)
    
    # 워크플로우 인스턴스 생성
    workflow = GoogleSearchWorkflow()
    
    # 상태 초기화
    state = {
        "execution_id": "verify-gelab-001",
        "workflow_id": "google-search",
        "parameters": {
            "query": "가방",
            "language": "ko",
            "num_results": 3
        },
        "node_logs": {}
    }
    
    # 실행
    try:
        # 1. search_google 노드 실행
        print("\n[Step 1] Executing 'search_google' node...")
        result = await workflow.execute_node("search_google", state)
        
        if result.success:
            print("\n[SUCCESS] Search completed successfully!")
            print(f"Message: {result.data.get('message')}")
            print(f"Stats: {result.data.get('rounds')} rounds")
        else:
            print(f"\n[FAILURE] Search failed: {result.error}")
            
    except Exception as e:
        print(f"\n[EXCEPTION] Workflow execution errored: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # asyncio 실행
    asyncio.run(run_workflow())
