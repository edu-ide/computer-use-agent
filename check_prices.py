
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
    if '할인 가격' in df.columns:
        print("Name | Original Price | Discount Price")
        for idx, row in df.iterrows():
            print(f"{row['상품명'][:20]}... | {row.get('원래 가격')} | {row.get('할인 가격')}")
    else:
        print("'할인 가격' column not found")
