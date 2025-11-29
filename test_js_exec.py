import time
from cua2_core.services.local_desktop import LocalDesktop

def test_js_execution():
    print("Initializing LocalDesktop...")
    desktop = LocalDesktop(display_num=98)  # Use different display to avoid conflict if possible
    
    print("Opening Google...")
    desktop.open("https://www.google.com")
    time.sleep(5)  # Wait for Chrome to launch and load
    
    print("Executing JS: document.title")
    title = desktop.evaluate_script("document.title")
    print(f"Result: {title}")
    
    print("Executing JS: 1 + 1")
    calc = desktop.evaluate_script("1 + 1")
    print(f"Result: {calc}")
    
    if "Google" in title:
        print("SUCCESS: Title retrieved correctly")
    else:
        print("FAILURE: Title mismatch")

if __name__ == "__main__":
    test_js_execution()
