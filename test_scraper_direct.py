
import asyncio
import sys
import os

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
        print(f"[Socket] {event}: {data}")

async def main():
    print("Starting direct scraper test...")
    scraper = CoupangScraper(MockSocket())
    
    # Params to skip details
    search_params = {
        'query': '여행가방',
        'headless': True,
        'collect_details': False, # Explicitly False
        'collect_similar': False,
        'max_results': 5,
        'use_existing_browser': False # Use new browser for test
    }
    
    try:
        results = await scraper.scrape(search_params)
        print("\n=== Scraper Results ===")
        print(f"Type: {type(results)}")
        if isinstance(results, dict):
            main_res = results.get('main_results', [])
            print(f"Main results count: {len(main_res)}")
            if main_res:
                print(f"First item: {main_res[0]}")
        else:
            print(f"Unexpected result: {results}")
            
    except Exception as e:
        print(f"Scraper failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
