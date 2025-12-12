import asyncio
import base64
import io
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
    print("Reproducing Fara-7B 400 Bad Request issue (Round 2)...")
    
    # 1. Create a dummy LARGE screenshot (1920x1080) to simulate real scenario
    img = Image.new('RGB', (1920, 1080), color='blue')
    
    # 2. Apply the FIX logic: Calculate size and resize
    print(f"Original image size: {img.size}")
    
    try:
        # Use the SAME config as in web_surfer.py
        system_prompt_data = get_computer_use_system_prompt(
            image=img,
            processor_im_cfg={
                "min_pixels": 3136,
                "max_pixels": 768 * 768, # 589824 pixels
                "patch_size": 14,
                "merge_size": 2,
            },
        )
        
        target_w, target_h = system_prompt_data["im_size"]
        print(f"Target resize dimensions: {target_w}x{target_h}")
        
        # Resize
        img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        print(f"Resized image size: {img_resized.size}")
        
        # Encode
        buffered = BytesIO()
        img_resized.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Extract system prompt
        raw_content = system_prompt_data.get("conversation", [{}])[0].get("content", "")
        system_prompt = ""
        if isinstance(raw_content, list):
            for item in raw_content:
                if isinstance(item, dict) and "text" in item:
                    system_prompt += item["text"]
        elif isinstance(raw_content, str):
            system_prompt = raw_content
            
        print(f"System prompt len: {len(system_prompt)}")
        
    except Exception as e:
        print(f"Error during prep: {e}")
        return

    # 3. Construct payload
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "Task: Search for 'test'\n\nHere is the current screenshot."},
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
    
    # 4. Send request
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
                # print(response.json())
                
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(reproduce_issue())
