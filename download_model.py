
from huggingface_hub import snapshot_download
import os

model_id = "stepfun-ai/GELab-Zero-4B-preview"
local_dir = "/mnt/sda1/models/llm/GELab-Zero-4B-preview"

print(f"Downloading {model_id} to {local_dir}...")
snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
print("Download complete.")
