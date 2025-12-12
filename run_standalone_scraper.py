
import asyncio
import sys
import os
import pandas as pd
from datetime import datetime

# Add project path
sys.path.append("/home/sk/ws/mcp-playwright/11_coupang_wing_web")

try:
    from scraper import CoupangScraper
except ImportError:
    print("Failed to import CoupangScraper")
    sys.exit(1)

# Mock socketio
class MockSocket:
    def emit(self, event, data):
        # Only print important events
        if event in ['log', 'result_update']:
             if isinstance(data, dict) and data.get('level') in ['info', 'success', 'error', 'warning']:
                 print(f"[{data.get('level', 'INFO').upper()}] {data.get('message', '')}")
             elif event == 'result_update':
                 pass # Too verbose

async def main():
    print("Starting standalone scraper...")
    scraper = CoupangScraper(MockSocket())
    
    # Read keywords
    KEYWORDS_FILE = "/home/sk/ws/mcp-playwright/computer-use-agent/keywords_to_search.xlsx"
    if not os.path.exists(KEYWORDS_FILE):
        print("Keywords file not found")
        return

    df = pd.read_excel(KEYWORDS_FILE)
    targets = df[df['Search'].astype(str).str.lower() == 'o']
    
    if targets.empty:
        print("No targets found in Excel (Search='o')")
        return

    all_results = []
    
    for idx, row in targets.iterrows():
        keyword = row['Keyword']
        print(f"\nProcessing keyword: {keyword}")
        
        search_params = {
            'query': keyword,
            'headless': True,
            'collect_details': False, # Important: Skip details
            'collect_similar': False,
            'max_results': 5,
            'use_existing_browser': False
        }
        
        try:
            result = await scraper.scrape(search_params)
            main_items = result.get('main_results', [])
            
            for item in main_items:
                # Map to required columns
                all_results.append({
                    '키워드': keyword,
                    '순위': item.get('rank'),
                    '상품명': item.get('name'),
                    '원래 가격': item.get('original_price'),
                    '할인 가격': item.get('price'),
                    '배송타입': item.get('seller_type'),
                    '평점': item.get('rating'),
                    '리뷰수': item.get('review_count'),
                    '썸네일 링크': item.get('thumbnail'),
                    '상품 URL': item.get('url'),
                    '썸네일': '' # Placeholder
                })
                
        except Exception as e:
            print(f"Error processing {keyword}: {e}")

    # Save to Excel
    if all_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = "/home/sk/ws/mcp-playwright/computer-use-agent/results"
        os.makedirs(save_dir, exist_ok=True)
        # Use first keyword for filename
        first_kwd = targets.iloc[0]['Keyword']
        filename = f"coupang_results_{first_kwd}_{timestamp}.xlsx"
        filepath = os.path.join(save_dir, filename)
        
        result_df = pd.DataFrame(all_results)
        # Reorder columns
        columns = ['키워드', '순위', '상품명', '원래 가격', '할인 가격', '배송타입', '평점', '리뷰수', '썸네일 링크', '상품 URL', '썸네일']
        # Add missing columns if any
        for col in columns:
            if col not in result_df.columns:
                result_df[col] = ''
        result_df = result_df[columns]
        
        result_df.to_excel(filepath, index=False)
        print(f"\nSaved {len(all_results)} items to: {filepath}")
    else:
        print("\nNo results collected.")

if __name__ == "__main__":
    asyncio.run(main())
