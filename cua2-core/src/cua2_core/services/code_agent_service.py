"""
Code Agent Service - HTML 분석 및 JavaScript 생성 전문 에이전트

역할:
- HTML 구조 분석
- 데이터 추출을 위한 JavaScript 코드 생성
- JavaScript 실행 및 결과 반환

모델: Orchestrator-8B (localhost:8081)
"""

import json
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class CodeAgentService:
    """
    HTML 분석 및 데이터 추출 JavaScript 생성을 담당하는 Code Agent
    
    VLM Agent와 역할 분리:
    - VLM Agent: 화면 관찰, 클릭/타이핑 등 인터랙션
    - Code Agent: HTML 분석, JavaScript 생성/실행, 데이터 추출
    """
    
    # Orchestrator-8B 서버 설정
    ORCHESTRATOR_API_URL = "http://localhost:8081/v1/chat/completions"
    ORCHESTRATOR_TIMEOUT = 30.0  # 30초 타임아웃 (코드 생성은 시간이 걸릴 수 있음)
    
    def __init__(self):
        self._http_client: Optional[httpx.AsyncClient] = None
        logger.info("[CodeAgent] 초기화 완료")
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 반환 (lazy init)"""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.ORCHESTRATOR_TIMEOUT)
        return self._http_client
    
    async def extract_data(
        self,
        desktop: Any,  # LocalDesktop 인스턴스
        task_description: str,
    ) -> Dict[str, Any]:
        """
        데이터 추출 메인 메서드
        
        Args:
            desktop: LocalDesktop 인스턴스 (evaluate_script 메서드 제공)
            task_description: 추출 작업 설명 (예: "로켓배송 아닌 상품 정보 추출")
        
        Returns:
            추출된 데이터 (dict)
        """
        logger.info(f"[CodeAgent] 데이터 추출 시작: {task_description}")
        
        # 1. HTML 샘플 수집
        html_sample = await self._sample_html(desktop)
        if not html_sample or html_sample.startswith("Error"):
            logger.error(f"[CodeAgent] HTML 샘플링 실패: {html_sample}")
            return {"error": "HTML 샘플링 실패", "details": html_sample}
        
        logger.info(f"[CodeAgent] HTML 샘플 수집 완료 ({len(html_sample)} chars)")
        
        # 2. Orchestrator-8B로 JavaScript 생성
        js_code = await self._generate_extraction_js(html_sample, task_description)
        if not js_code or js_code.startswith("Error"):
            logger.error(f"[CodeAgent] JS 생성 실패: {js_code}")
            return {"error": "JS 생성 실패", "details": js_code}
        
        logger.info(f"[CodeAgent] JavaScript 생성 완료:\n{js_code[:200]}...")
        
        # 3. 생성된 JavaScript 실행
        result = desktop.evaluate_script(js_code)
        
        # 4. 결과 파싱
        if result.startswith("Error") or result.startswith("JS Exception"):
            logger.error(f"[CodeAgent] JS 실행 실패: {result}")
            return {"error": "JS 실행 실패", "details": result}
        
        try:
            data = json.loads(result)
            logger.info(f"[CodeAgent] 데이터 추출 성공: {len(data) if isinstance(data, list) else 1}개 항목")
            return {"success": True, "data": data}
        except json.JSONDecodeError as e:
            logger.error(f"[CodeAgent] JSON 파싱 실패: {e}")
            return {"error": "JSON 파싱 실패", "details": str(e), "raw": result[:500]}
    
    async def _sample_html(self, desktop: Any) -> str:
        """
        현재 페이지의 HTML 샘플 수집
        
        Args:
            desktop: LocalDesktop 인스턴스
        
        Returns:
            HTML 샘플 (첫 5000자)
        """
        js_code = "document.body.outerHTML.slice(0, 5000)"
        result = desktop.evaluate_script(js_code)
        return result
    
    async def _generate_extraction_js(
        self,
        html_sample: str,
        task_description: str,
    ) -> str:
        """
        Orchestrator-8B를 사용하여 데이터 추출 JavaScript 생성
        
        Args:
            html_sample: HTML 샘플
            task_description: 추출 작업 설명
        
        Returns:
            생성된 JavaScript 코드
        """
        system_prompt = """You are a JavaScript expert. Generate JavaScript code to extract data from HTML.

**RULES:**
1. Analyze the provided HTML structure carefully
2. Generate a JavaScript IIFE that extracts the requested data
3. Return ONLY the JavaScript code, no explanations
4. The code must return a JSON-serializable result (array or object)
5. Use querySelector/querySelectorAll with correct selectors based on actual HTML
6. Handle null/undefined cases gracefully

**OUTPUT FORMAT:**
```javascript
(() => {
    // Your extraction logic here
    return extractedData;
})()
```
"""
        
        user_content = f"""**Task:** {task_description}

**HTML Sample:**
```html
{html_sample}
```

Generate JavaScript code to extract the data. Return ONLY the code, no markdown formatting."""
        
        try:
            client = await self._get_http_client()
            response = await client.post(
                self.ORCHESTRATOR_API_URL,
                json={
                    "model": "orchestrator-8b",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2048,
                },
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # 코드 블록 제거 (```javascript ... ``` 형식)
            if "```" in content:
                # ``` 사이의 코드만 추출
                parts = content.split("```")
                if len(parts) >= 3:
                    code = parts[1]
                    # javascript 키워드 제거
                    if code.startswith("javascript"):
                        code = code[10:].strip()
                    elif code.startswith("js"):
                        code = code[2:].strip()
                    return code.strip()
            
            return content.strip()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"[CodeAgent] Orchestrator-8B HTTP 오류: {e}")
            return f"Error: HTTP {e.response.status_code}"
        except httpx.TimeoutException:
            logger.error("[CodeAgent] Orchestrator-8B 타임아웃")
            return "Error: Timeout"
        except Exception as e:
            logger.error(f"[CodeAgent] Orchestrator-8B 호출 실패: {e}")
            return f"Error: {str(e)}"


# 싱글톤 인스턴스
_code_agent_instance: Optional[CodeAgentService] = None


def get_code_agent() -> CodeAgentService:
    """Code Agent 싱글톤 인스턴스 반환"""
    global _code_agent_instance
    if _code_agent_instance is None:
        _code_agent_instance = CodeAgentService()
    return _code_agent_instance
