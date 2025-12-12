from cua2_core.services.headless_desktop import HeadlessDesktop
import time

def test_input():
    print("Initializing HeadlessDesktop...")
    desktop = HeadlessDesktop()
    
    print("\n[Test 1] Open Google")
    desktop.open("https://www.google.co.kr")
    time.sleep(5)
    
    print("\n[Test 2] Click Input (Center)")
    # Google Search is approx center 960, 540 (or slightly above)
    # Using 960, 480 as rough estimate for 1080p
    desktop.move_mouse(960, 480)
    desktop.left_click()
    time.sleep(1)
    
    print("\n[Test 3] Type 'test search'")
    desktop.write("test search")
    time.sleep(1)
    
    print("\n[Test 4] Press Enter")
    desktop.press(['Enter'])
    time.sleep(5)
    
    print("\n[Test 5] Screenshot & Check")
    img = desktop.screenshot()
    img.save("/tmp/vlm_agent_runner/debug_input_test.png")
    print("Saved screenshot to /tmp/vlm_agent_runner/debug_input_test.png")
    
    desktop.kill()

if __name__ == "__main__":
    test_input()
