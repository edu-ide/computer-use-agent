
import pandas as pd
import os

RESULT_FILE = "/home/sk/ws/mcp-playwright/computer-use-agent/results/coupang_results_전동빗자루_20251212_160043.xlsx"

if os.path.exists(RESULT_FILE):
    df = pd.read_excel(RESULT_FILE)
    print(f"Columns: {df.columns.tolist()}")
    if not df.empty:
        print("First row example:")
        print(df.iloc[0].to_dict())
else:
    print("Result file not found.")
