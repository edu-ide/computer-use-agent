
import sys
import os
import pandas as pd
from datetime import datetime

# Add project path
sys.path.append("/home/sk/ws/mcp-playwright/11_coupang_wing_web")
sys.path.append("/home/sk/ws/mcp-playwright/computer-use-agent/cua2-core/src")

try:
    from scraper import CoupangScraper
    print("✅ CoupangScraper imported successfully")
except ImportError as e:
    print(f"❌ Failed to import CoupangScraper: {e}")
    sys.exit(1)

KEYWORDS_FILE = "/home/sk/ws/mcp-playwright/computer-use-agent/keywords_to_search.xlsx"

def test_excel():
    print(f"Testing Excel file: {KEYWORDS_FILE}")
    if not os.path.exists(KEYWORDS_FILE):
        print(f"❌ File not found: {KEYWORDS_FILE}")
        return

    try:
        df = pd.read_excel(KEYWORDS_FILE)
        print(f"✅ Excel read successfully. Columns: {df.columns.tolist()}")
        print(f"   Rows: {len(df)}")
        
        if 'Search' in df.columns:
            targets = df[df['Search'].astype(str).str.lower() == 'o']
            print(f"   Targets found (Search='o'): {len(targets)}")
            for idx, row in targets.iterrows():
                print(f"     - {row.get('Keyword')} (Count: {row.get('Count')})")
        else:
            print("   'Search' column not found.")
            
    except Exception as e:
        print(f"❌ Failed to read Excel: {e}")

if __name__ == "__main__":
    test_excel()
