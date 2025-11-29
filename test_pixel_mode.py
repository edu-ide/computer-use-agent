import sys
import os
import time

# Add src to path
sys.path.append("/home/sk/ws/mcp-playwright/computer-use-agent/cua2-core/src")

from cua2_core.services.local_desktop import LocalDesktop
from cua2_core.services.agent_utils.local_desktop_agent import LocalVisionAgent, SanitizedExecutorProxy
from smolagents import Model, LocalPythonExecutor

# Dummy model
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
agent = LocalVisionAgent(model=DummyModel(), data_dir="/tmp/test_agent_pixels", desktop=desktop)

# Use the agent's executor which is already wrapped with SanitizedExecutorProxy
proxy = agent.python_executor

print("\n--- Testing click(500, 100) WITHOUT # pixels ---")
# This should use normalized coordinates (500/1000 -> 640, 100/1000 -> 72)
code_normal = 'click(500, 100)'
try:
    proxy(code_normal)
except Exception as e:
    print(f"Error: {e}")

print("\n--- Testing click(500, 100) WITH # pixels ---")
# This should use pixel coordinates (500, 100)
code_pixels = 'click(500, 100) # pixels'
try:
    proxy(code_pixels)
except Exception as e:
    print(f"Error: {e}")

# Clean up
desktop.kill()
