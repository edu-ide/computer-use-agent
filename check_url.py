
import pandas as pd
import glob
import os

# Find the latest result file for '크로스백'
files = glob.glob("/home/sk/ws/mcp-playwright/computer-use-agent/results/coupang_results_크로스백_*.xlsx")
if not files:
    print("No file found")
else:
    latest_file = max(files, key=os.path.getctime)
    print(f"Reading: {latest_file}")
    
    df = pd.read_excel(latest_file)
    if '상품 URL' in df.columns:
        urls = df['상품 URL'].dropna().tolist()
        print(f"Found {len(urls)} URLs.")
        for i, url in enumerate(urls[:3]):
            print(f"URL {i+1}: {url}")
    else:
        print("'상품 URL' column not found")
