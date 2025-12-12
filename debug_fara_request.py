import asyncio
import base64
import json
import logging
import os
import sys
from io import BytesIO

import httpx
from PIL import Image

# Add src to python path
sys.path.append(os.path.join(os.getcwd(), "cua2-core/src"))

from cua2_core.fara.prompts import get_computer_use_system_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def reproduce_issue():
    print("Reproducing Fara-7B 400 Bad Request issue...")
    
    # 1. Create a dummy screenshot (768x768)
    img = Image.new('RGB', (768, 768), color='red')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # 2. Generate system prompt using the ACTUAL function from codebase
    # This is where the complexity likely lies (NousFnCallPrompt)
    print("Generating system prompt...")
    try:
        system_prompt_data = get_computer_use_system_prompt(
            image=img,
            processor_im_cfg={
                "min_pixels": 3136,
                "max_pixels": 512 * 512,
                "patch_size": 14,
                "merge_size": 2,
            },
        )
        
        # Extract content logic from web_surfer.py
        raw_content = system_prompt_data.get("conversation", [{}])[0].get("content", "")
        system_prompt = ""
        if isinstance(raw_content, list):
            for item in raw_content:
                if isinstance(item, dict) and "text" in item:
                    system_prompt += item["text"]
        elif isinstance(raw_content, str):
            system_prompt = raw_content
            
        print(f"System prompt generated (len: {len(system_prompt)})")
        # print(f"System prompt preview: {system_prompt[:200]}...")
        
    except Exception as e:
        print(f"Error generating system prompt: {e}")
        return

    # 3. Construct the payload exactly like web_surfer.py
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "Task: Go to google.com\n\nHere is the current screenshot. What action should I take next?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]
        }
    ]
    
    payload = {
        "model": "Fara-7B",
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    
    # 4. Send request using httpx
    print("Sending request to localhost:30001...")
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            response = await client.post(
                "http://localhost:30001/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            print(f"Status Code: {response.status_code}")
            if response.status_code != 200:
                print(f"Response Text: {response.text}")
            else:
                print("Success!")
                print(response.json())
                
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(reproduce_issue())
