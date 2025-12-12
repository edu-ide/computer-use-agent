
import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.append("/mnt/sda1/projects/computer-use-agent/cua2-core/src")

from cua2_core.fara.web_surfer import FaraWebSurfer, WebSurferConfig
from cua2_core.fara.playwright_controller import PlaywrightController

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_playwright_typing():
    """Test if Playwright can type '가방' correctly."""
    print("\n[Test 1] Testing Playwright Typing...")
    controller = PlaywrightController(viewport_width=1280, viewport_height=720)
    
    try:
        await controller.start()
        await controller.visit_url("https://www.google.com")
        await asyncio.sleep(2)
        
        # Click search box (approximate coordinates for Google home)
        # Usually center-ish 
        await controller._click_coords_stateful(640, 360) 
        await asyncio.sleep(1)
        
        target_text = "가방"
        print(f"Typing '{target_text}'...")
        await controller.type_text(target_text)
        await asyncio.sleep(1)
        
        # Capture value to verify
        val = await controller._page.evaluate("document.activeElement.value")
        print(f"Input value: {val}")
        
        if val == target_text:
            print("SUCCESS: Typed text matches.")
        else:
            print(f"FAILURE: Typed text mismatch. Expected '{target_text}', got '{val}'")
            
    except Exception as e:
        print(f"Error in typing test: {e}")
    finally:
        await controller.close()

async def test_full_agent():
    """Test full FaraWebSurfer agent loop."""
    print("\n[Test 2] Testing Full Agent Loop...")
    
    config = WebSurferConfig(
        start_page="https://www.google.com",
        max_rounds=5
    )
    
    agent = FaraWebSurfer(
        config=config,
        llm_base_url="http://localhost:30001/v1",
        llm_model="/mnt/sda1/models/llm/GELab-Zero-4B-preview" 
    )
    
    try:
        await agent.initialize()
        
        task = "검색창에 '가방'을 입력하고 Enter를 눌러 검색하세요."
        print(f"Executing task: {task}")
        
        result = await agent.execute_task(task)
        
        print("Agent execution result:")
        print(result)
        
    except Exception as e:
        print(f"Error in agent test: {e}")
    finally:
        await agent.close()

async def main():
    await test_playwright_typing()
    # Uncomment to run full agent test
    await test_full_agent()

if __name__ == "__main__":
    asyncio.run(main())
