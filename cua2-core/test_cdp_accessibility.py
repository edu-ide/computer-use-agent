#!/usr/bin/env python3
"""
CDP Accessibility Tree 테스트 스크립트
Playwright 사용하여 CDP Accessibility API 테스트
"""

import asyncio
import json
from playwright.async_api import async_playwright


async def test_cdp_accessibility():
    """CDP를 통한 Accessibility Tree 가져오기 테스트"""

    async with async_playwright() as p:
        # 브라우저 실행 (headless 모드로 테스트)
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"]
        )

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )

        page = await context.new_page()

        # CDP 세션 생성
        cdp_session = await context.new_cdp_session(page)

        # 테스트 페이지로 이동 (Google 검색)
        print("페이지 로딩 중...")
        await page.goto("https://www.google.com")
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(2)

        print("\n" + "="*60)
        print("CDP Accessibility Tree 테스트")
        print("="*60)

        # 1. Accessibility 도메인 활성화
        print("\n[1] Accessibility 도메인 활성화...")
        try:
            await cdp_session.send("Accessibility.enable")
            print("    ✅ Accessibility.enable 성공")
        except Exception as e:
            print(f"    ❌ Accessibility.enable 실패: {e}")
            return

        # 2. Full Accessibility Tree 가져오기
        print("\n[2] Full Accessibility Tree 가져오기...")
        try:
            result = await cdp_session.send("Accessibility.getFullAXTree")
            nodes = result.get("nodes", [])
            print(f"    ✅ 총 {len(nodes)}개의 노드 발견")

            # 몇 개의 노드 샘플 출력
            print("\n    샘플 노드 (처음 10개):")
            for i, node in enumerate(nodes[:10]):
                name = node.get("name", {}).get("value", "")
                role = node.get("role", {}).get("value", "")
                print(f"      [{i}] role={role}, name={name[:50] if name else ''}")

        except Exception as e:
            print(f"    ❌ getFullAXTree 실패: {e}")

            # 대안: getRootAXNode 시도
            print("\n    대안으로 getRootAXNode 시도...")
            try:
                result = await cdp_session.send("Accessibility.getRootAXNode")
                print(f"    ✅ getRootAXNode 성공: {json.dumps(result, indent=2)[:500]}")
            except Exception as e2:
                print(f"    ❌ getRootAXNode도 실패: {e2}")

        # 3. 인터랙티브 요소만 필터링
        print("\n[3] 인터랙티브 요소 필터링...")
        interactive_roles = [
            "button", "link", "textbox", "searchbox", "combobox",
            "checkbox", "radio", "menuitem", "tab", "listitem"
        ]

        try:
            result = await cdp_session.send("Accessibility.getFullAXTree")
            nodes = result.get("nodes", [])

            interactive_nodes = []
            for node in nodes:
                role = node.get("role", {}).get("value", "")
                if role in interactive_roles:
                    name = node.get("name", {}).get("value", "")
                    interactive_nodes.append({
                        "role": role,
                        "name": name[:100] if name else "",
                        "nodeId": node.get("nodeId", "")
                    })

            print(f"    ✅ {len(interactive_nodes)}개의 인터랙티브 요소 발견")
            print("\n    인터랙티브 요소 목록:")
            for i, node in enumerate(interactive_nodes[:20]):
                print(f"      [{i}] {node['role']}: {node['name']}")

        except Exception as e:
            print(f"    ❌ 인터랙티브 요소 필터링 실패: {e}")

        # 4. 특정 요소의 위치 정보 가져오기 (DOM.getBoxModel)
        print("\n[4] DOM 노드의 위치 정보 테스트...")
        try:
            # 검색창 찾기
            search_input = await page.query_selector('textarea[name="q"], input[name="q"]')
            if search_input:
                # 요소의 bounding box
                box = await search_input.bounding_box()
                print(f"    ✅ 검색창 위치: x={box['x']:.0f}, y={box['y']:.0f}, w={box['width']:.0f}, h={box['height']:.0f}")
            else:
                print("    ⚠️ 검색창을 찾을 수 없음")
        except Exception as e:
            print(f"    ❌ 위치 정보 가져오기 실패: {e}")

        # 5. queryAXTree 테스트 (특정 역할의 노드 검색)
        print("\n[5] queryAXTree 테스트 (button 역할 검색)...")
        try:
            # 먼저 root 노드 ID 필요
            root_result = await cdp_session.send("Accessibility.getRootAXNode")
            root_id = root_result.get("node", {}).get("nodeId")

            if root_id:
                query_result = await cdp_session.send("Accessibility.queryAXTree", {
                    "nodeId": root_id,
                    "role": "button"
                })
                button_nodes = query_result.get("nodes", [])
                print(f"    ✅ {len(button_nodes)}개의 버튼 발견")
                for i, node in enumerate(button_nodes[:5]):
                    name = node.get("name", {}).get("value", "")
                    print(f"      [{i}] button: {name}")
        except Exception as e:
            print(f"    ❌ queryAXTree 실패: {e}")

        print("\n" + "="*60)
        print("테스트 완료!")
        print("="*60)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(test_cdp_accessibility())
