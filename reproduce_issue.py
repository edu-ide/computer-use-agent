import sys
import os
import time

# Add src to path
sys.path.append("/home/sk/ws/mcp-playwright/computer-use-agent/cua2-core/src")

from cua2_core.services.local_desktop import LocalDesktop
from cua2_core.services.agent_utils.local_desktop_agent import LocalVisionAgent
from smolagents import Model

# Dummy model to satisfy CodeAgent requirement
class DummyModel(Model):
    def __init__(self):
        super().__init__()
    def __call__(self, *args, **kwargs):
        return "final_answer('done')"
        
    def generate_stream(self, messages):
        yield "final_answer('done')"

print("Initializing LocalDesktop...")
desktop = LocalDesktop(width=1280, height=720)

print("Initializing LocalVisionAgent...")
agent = LocalVisionAgent(model=DummyModel(), data_dir="/tmp/test_agent", desktop=desktop)

print("\n--- Testing click(300, 100) ---")
# This should trigger the logging we added
try:
    result = agent.tools["click"](300, 100)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

print("\n--- Testing click(500, 500) ---")
try:
    result = agent.tools["click"](500, 500)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

# Clean up
desktop.kill()
