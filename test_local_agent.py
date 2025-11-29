#!/usr/bin/env python3
"""
로컬 에이전트 테스트 스크립트
llama-server가 실행 중이어야 합니다.
"""

import sys
sys.path.insert(0, "cua2-core/src")

from cua2_core.services.local_desktop import LocalDesktop
from cua2_core.services.agent_utils.get_model import get_model

def test_local_desktop():
    """로컬 데스크톱 테스트"""
    print("=== 로컬 데스크톱 테스트 ===")
    try:
        desktop = LocalDesktop()
        width, height = desktop.get_screen_size()
        print(f"화면 크기: {width}x{height}")

        # 스크린샷 테스트
        screenshot = desktop.screenshot()
        screenshot.save("/tmp/test_screenshot.png")
        print(f"스크린샷 저장: /tmp/test_screenshot.png")
        print("로컬 데스크톱 테스트 성공!")
        return True
    except Exception as e:
        print(f"로컬 데스크톱 테스트 실패: {e}")
        return False

def test_model_connection():
    """모델 연결 테스트"""
    print("\n=== 모델 연결 테스트 ===")
    try:
        model = get_model("local-qwen3-vl")
        print(f"모델 타입: {type(model)}")
        print("모델 연결 테스트 성공!")
        return True
    except Exception as e:
        print(f"모델 연결 테스트 실패: {e}")
        return False

def test_simple_agent():
    """간단한 에이전트 테스트"""
    print("\n=== 간단한 에이전트 테스트 ===")
    try:
        from cua2_core.services.local_desktop import LocalDesktop
        from cua2_core.services.agent_utils.local_desktop_agent import LocalVisionAgent

        desktop = LocalDesktop()
        model = get_model("local-qwen3-vl")

        agent = LocalVisionAgent(
            model=model,
            data_dir="/tmp/agent_test",
            desktop=desktop,
            max_steps=3,
        )

        print(f"에이전트 생성 성공!")
        print(f"사용 가능한 도구: {list(agent.tools.keys())}")
        return True
    except Exception as e:
        import traceback
        print(f"에이전트 테스트 실패: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    results = []

    results.append(("로컬 데스크톱", test_local_desktop()))
    results.append(("모델 연결", test_model_connection()))
    results.append(("에이전트 생성", test_simple_agent()))

    print("\n" + "=" * 40)
    print("테스트 결과:")
    for name, passed in results:
        status = "✓ 성공" if passed else "✗ 실패"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    sys.exit(0 if all_passed else 1)
