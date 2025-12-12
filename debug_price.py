
from bs4 import BeautifulSoup
import re
import os

HTML_FILE = "/home/sk/ws/mcp-playwright/computer-use-agent/saved_html/search_크로스백_20251212_214312.html"

def debug_price():
    if not os.path.exists(HTML_FILE):
        print(f"File not found: {HTML_FILE}")
        return

    with open(HTML_FILE, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Logic from scraper.py
    products = soup.find_all('li', class_=lambda x: x and 'ProductUnit_productUnit' in x)
    if not products:
        products = soup.find_all('li', class_='search-product')
        print("Using OLD UI selectors")
    else:
        print("Using NEW UI selectors")

    print(f"Found {len(products)} products.")

    target_found = False
    for i, product in enumerate(products):
        # Name
        name_elem = product.find(class_=lambda x: x and 'productNameV2' in str(x))
        if not name_elem:
            name_elem = product.find(class_='name')
        name = name_elem.get_text(strip=True) if name_elem else "No Name"
        
        if "담앤드솜" in name:
            print(f"\n--- MATCHED Product {i+1}: {name} ---")
            target_found = True
        elif i < 3:
             print(f"\n--- Product {i+1}: {name} ---")
        elif not target_found:
             continue
        
        if target_found or i < 3:
             # Price Debugging
             price_area = product.find(class_=lambda x: x and 'PriceArea' in str(x))
             if price_area:
                 print("Found PriceArea")
                 candidates = price_area.find_all(class_=lambda x: x and 'fw-font-bold' in str(x))
                 found = False
                 for cand in candidates:
                     text = cand.get_text(strip=True)
                     print(f"Candidate: '{text}'")
                     if '%' in text:
                         print("  -> Skipped (%)")
                         continue
                     if not re.search(r'\d', text):
                         print("  -> Skipped (no digits)")
                         continue

                     if ':' in text or '남음' in text or '도착' in text:
                         print(f"  -> Skipped (time/date/etc): {text}")
                         continue
                     
                     clean = text.replace(',', '').replace('원', '').replace('₩', '').strip()
                     match = re.search(r'\d+', clean)
                     if match:
                         val = int(match.group())
                         print(f"  -> MATCHED PRICE: {val}")
                         found = True
                         break
                 
                 if not found:
                     print("  -> No price matched in loop.")
     
             else:
                 print("No PriceArea found (Old UI or different structure)")
                 price_elem = product.find(class_=lambda x: x and 'price-value' in str(x))
                 if price_elem:
                      print(f"Found price-value: {price_elem.get_text(strip=True)}")

        # Extract Price (Replicating logical flow)
        price = 0
        original_price = 0
        
        # New UI
        if price_area:
            # Try finding price-value inside PriceArea
            pv = price_area.find(class_=lambda x: x and 'priceValue' in str(x))
            if pv:
                 print(f"Alternative: found 'priceValue' class: {pv.get_text(strip=True)}")

            price_div = price_area.find(class_=lambda x: x and 'fw-font-bold' in str(x)) # Loosened selector
            if price_div:
                 print(f"Alternative: found 'fw-font-bold': {price_div.get_text(strip=True)}")

if __name__ == "__main__":
    debug_price()
