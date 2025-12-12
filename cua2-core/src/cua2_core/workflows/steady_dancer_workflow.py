# -*- coding: utf-8 -*-
"""
SteadyDancer 워크플로우 - AI 댄스 영상 생성

워크플로우 단계:
1. Flux2로 여성 이미지 생성
2. YouTube API로 트렌드 댄스 영상 검색
3. yt-dlp로 영상 다운로드
4. ComfyUI SteadyDancer로 추론
5. 결과 영상 저장
"""

import asyncio
import json
import os
import subprocess
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from .workflow_base import (
    WorkflowBase,
    WorkflowConfig,
    WorkflowNode,
    WorkflowState,
    NodeResult,
    VLMErrorType,
)


class SteadyDancerWorkflow(WorkflowBase):
    """
    SteadyDancer AI 댄스 영상 생성 워크플로우

    Flow:
    1. generate_image - Flux2로 여성 이미지 생성
    2. search_dance - YouTube API로 트렌드 댄스 쇼츠 검색
    3. download_video - yt-dlp로 영상 다운로드
    4. run_steadydancer - ComfyUI API로 SteadyDancer 추론
    5. complete - 완료
    """

    # ComfyUI SteadyDancer 워크플로우 템플릿 경로
    COMFYUI_WORKFLOW_PATH = "/mnt/sda1/Download/wanvideo_SteadyDancer_example_01.json"

    # 기본 설정
    COMFYUI_URL = "http://127.0.0.1:8188"
    FLUX2_MODEL_PATH = "/mnt/sda1/projects/flux2_model"
    OUTPUT_DIR = "/mnt/sda1/projects/computer-use-agent/outputs/steady_dancer"

    # YouTube API
    YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

    def __init__(self, agent_runner=None):
        super().__init__()
        self._agent_runner = agent_runner

        # 출력 디렉토리 생성
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    @property
    def config(self) -> WorkflowConfig:
        return WorkflowConfig(
            id="steady-dancer",
            name="SteadyDancer AI 댄스 영상",
            description="Flux2로 이미지를 생성하고, 트렌드 댄스 영상을 찾아 AI로 춤추는 영상을 만듭니다.",
            icon="MusicVideo",
            color="#e91e63",
            category="video",
            parameters=[
                {
                    "name": "image_prompt",
                    "type": "string",
                    "label": "이미지 프롬프트",
                    "placeholder": "예: beautiful korean woman, full body, white background, fashion model",
                    "required": True,
                },
                {
                    "name": "dance_keyword",
                    "type": "string",
                    "label": "댄스 검색 키워드",
                    "placeholder": "예: kpop dance, viral dance challenge",
                    "default": "trending dance shorts",
                },
                {
                    "name": "video_prompt",
                    "type": "string",
                    "label": "영상 프롬프트",
                    "placeholder": "예: woman dancing gracefully",
                    "default": "woman dancing, smooth movement",
                },
                {
                    "name": "num_frames",
                    "type": "number",
                    "label": "프레임 수",
                    "default": 81,
                    "min": 41,
                    "max": 176,
                },
                {
                    "name": "youtube_api_key",
                    "type": "string",
                    "label": "YouTube API 키",
                    "placeholder": "API 키 입력 (없으면 환경변수 사용)",
                    "required": False,
                },
                {
                    "name": "comfyui_url",
                    "type": "string",
                    "label": "ComfyUI URL",
                    "default": "http://127.0.0.1:8188",
                },
                {
                    "name": "use_existing_image",
                    "type": "string",
                    "label": "기존 이미지 경로 (선택)",
                    "placeholder": "이미지 경로 입력 시 Flux2 생성 스킵",
                    "required": False,
                },
                {
                    "name": "use_existing_video",
                    "type": "string",
                    "label": "기존 영상 경로 (선택)",
                    "placeholder": "영상 경로 입력 시 YouTube 검색 스킵",
                    "required": False,
                },
            ],
        )

    @property
    def nodes(self) -> List[WorkflowNode]:
        return [
            WorkflowNode(
                name="generate_image",
                display_name="이미지 생성",
                description="Flux2로 여성 이미지 생성",
                on_success="search_dance",
                on_failure="error_handler",
                timeout_sec=300,
            ),
            WorkflowNode(
                name="search_dance",
                display_name="댄스 영상 검색",
                description="YouTube API로 트렌드 댄스 쇼츠 검색",
                on_success="download_video",
                on_failure="error_handler",
                timeout_sec=60,
            ),
            WorkflowNode(
                name="download_video",
                display_name="영상 다운로드",
                description="yt-dlp로 댄스 영상 다운로드",
                on_success="run_steadydancer",
                on_failure="error_handler",
                timeout_sec=120,
            ),
            WorkflowNode(
                name="run_steadydancer",
                display_name="SteadyDancer 추론",
                description="ComfyUI SteadyDancer로 춤 영상 생성",
                on_success="complete",
                on_failure="error_handler",
                timeout_sec=1800,  # 30분 (영상 생성은 오래 걸림)
            ),
            WorkflowNode(
                name="complete",
                display_name="완료",
                description="워크플로우 완료",
            ),
            WorkflowNode(
                name="error_handler",
                display_name="에러 처리",
                description="에러 처리",
            ),
        ]

    @property
    def start_node(self) -> str:
        return "generate_image"

    async def execute_node(self, node_name: str, state: WorkflowState) -> NodeResult:
        """노드별 실행 로직"""
        handlers = {
            "generate_image": self._generate_image,
            "search_dance": self._search_dance,
            "download_video": self._download_video,
            "run_steadydancer": self._run_steadydancer,
            "complete": self._complete,
            "error_handler": self._error_handler,
        }

        handler = handlers.get(node_name)
        if handler:
            return await handler(state)

        return NodeResult(success=False, error=f"Unknown node: {node_name}")

    async def _generate_image(self, state: WorkflowState) -> NodeResult:
        """Flux2로 여성 이미지 생성"""
        try:
            parameters = state.get("parameters", {})

            # 기존 이미지 사용 옵션
            existing_image = parameters.get("use_existing_image", "")
            if existing_image and os.path.exists(existing_image):
                print(f"[SteadyDancer] 기존 이미지 사용: {existing_image}")
                return NodeResult(
                    success=True,
                    data={"generated_image_path": existing_image}
                )

            image_prompt = parameters.get("image_prompt", "beautiful woman, full body, white background")

            # 출력 경로
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.OUTPUT_DIR, f"flux2_image_{timestamp}.png")

            print(f"[SteadyDancer] Flux2 이미지 생성 시작: {image_prompt}")

            # Flux2 추론 실행 (subprocess로 별도 프로세스에서 실행)
            flux2_script = f'''
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "{self.FLUX2_MODEL_PATH}",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="{image_prompt}",
    generator=torch.Generator(device="cuda").manual_seed(42),
    num_inference_steps=28,
    guidance_scale=3.5,
    height=768,
    width=512,
).images[0]

image.save("{output_path}")
print("Image saved to {output_path}")
'''

            # 임시 스크립트 파일 생성
            script_path = os.path.join(self.OUTPUT_DIR, f"flux2_gen_{timestamp}.py")
            with open(script_path, "w") as f:
                f.write(flux2_script)

            # 실행
            process = await asyncio.create_subprocess_exec(
                "python", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            # 임시 스크립트 삭제
            if os.path.exists(script_path):
                os.remove(script_path)

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                return NodeResult(success=False, error=f"Flux2 생성 실패: {error_msg}")

            if not os.path.exists(output_path):
                return NodeResult(success=False, error="이미지 파일이 생성되지 않았습니다")

            print(f"[SteadyDancer] 이미지 생성 완료: {output_path}")

            return NodeResult(
                success=True,
                data={"generated_image_path": output_path}
            )

        except Exception as e:
            return NodeResult(success=False, error=str(e))

    async def _search_dance(self, state: WorkflowState) -> NodeResult:
        """YouTube API로 트렌드 댄스 쇼츠 검색"""
        try:
            parameters = state.get("parameters", {})

            # 기존 영상 사용 옵션
            existing_video = parameters.get("use_existing_video", "")
            if existing_video and os.path.exists(existing_video):
                print(f"[SteadyDancer] 기존 영상 사용: {existing_video}")
                return NodeResult(
                    success=True,
                    data={
                        "dance_video_url": None,
                        "dance_video_path": existing_video,
                        "video_title": "User provided video",
                    }
                )

            dance_keyword = parameters.get("dance_keyword", "trending dance shorts")
            api_key = parameters.get("youtube_api_key", "") or self.YOUTUBE_API_KEY

            if not api_key:
                return NodeResult(success=False, error="YouTube API 키가 필요합니다")

            print(f"[SteadyDancer] YouTube 검색: {dance_keyword}")

            # YouTube Data API v3 검색
            search_url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": dance_keyword,
                "type": "video",
                "videoDuration": "short",  # 쇼츠 (4분 미만)
                "order": "viewCount",  # 조회수 순
                "maxResults": 10,
                "key": api_key,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return NodeResult(success=False, error=f"YouTube API 오류: {error_text}")

                    data = await response.json()

            items = data.get("items", [])
            if not items:
                return NodeResult(success=False, error="검색 결과가 없습니다")

            # 첫 번째 결과 선택
            video = items[0]
            video_id = video["id"]["videoId"]
            video_title = video["snippet"]["title"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            print(f"[SteadyDancer] 선택된 영상: {video_title}")
            print(f"[SteadyDancer] URL: {video_url}")

            return NodeResult(
                success=True,
                data={
                    "dance_video_url": video_url,
                    "dance_video_id": video_id,
                    "video_title": video_title,
                    "search_results": [
                        {
                            "id": item["id"]["videoId"],
                            "title": item["snippet"]["title"],
                            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                        }
                        for item in items
                    ],
                }
            )

        except Exception as e:
            return NodeResult(success=False, error=str(e))

    async def _download_video(self, state: WorkflowState) -> NodeResult:
        """yt-dlp로 댄스 영상 다운로드"""
        try:
            data = state.get("data", {})

            # 이미 다운로드된 경로가 있으면 스킵
            if data.get("dance_video_path"):
                return NodeResult(
                    success=True,
                    data={"dance_video_path": data["dance_video_path"]}
                )

            video_url = data.get("dance_video_url")
            if not video_url:
                return NodeResult(success=False, error="다운로드할 영상 URL이 없습니다")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.OUTPUT_DIR, f"dance_video_{timestamp}.mp4")

            print(f"[SteadyDancer] 영상 다운로드 중: {video_url}")

            # yt-dlp로 다운로드
            process = await asyncio.create_subprocess_exec(
                "yt-dlp",
                "-f", "best[height<=720]",  # 720p 이하
                "-o", output_path,
                video_url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                return NodeResult(success=False, error=f"다운로드 실패: {error_msg}")

            # yt-dlp가 확장자를 변경할 수 있으므로 실제 파일 찾기
            if not os.path.exists(output_path):
                # 비슷한 이름의 파일 찾기
                base_name = os.path.splitext(output_path)[0]
                for ext in [".mp4", ".webm", ".mkv"]:
                    if os.path.exists(base_name + ext):
                        output_path = base_name + ext
                        break

            if not os.path.exists(output_path):
                return NodeResult(success=False, error="다운로드된 파일을 찾을 수 없습니다")

            print(f"[SteadyDancer] 다운로드 완료: {output_path}")

            return NodeResult(
                success=True,
                data={"dance_video_path": output_path}
            )

        except Exception as e:
            return NodeResult(success=False, error=str(e))

    async def _run_steadydancer(self, state: WorkflowState) -> NodeResult:
        """ComfyUI SteadyDancer로 춤 영상 생성"""
        try:
            parameters = state.get("parameters", {})
            data = state.get("data", {})

            image_path = data.get("generated_image_path")
            video_path = data.get("dance_video_path")

            if not image_path or not os.path.exists(image_path):
                return NodeResult(success=False, error="이미지 파일이 없습니다")

            if not video_path or not os.path.exists(video_path):
                return NodeResult(success=False, error="댄스 영상 파일이 없습니다")

            comfyui_url = parameters.get("comfyui_url", self.COMFYUI_URL)
            video_prompt = parameters.get("video_prompt", "woman dancing, smooth movement")
            num_frames = parameters.get("num_frames", 81)

            print(f"[SteadyDancer] ComfyUI 추론 시작")
            print(f"  - 이미지: {image_path}")
            print(f"  - 영상: {video_path}")
            print(f"  - 프롬프트: {video_prompt}")

            # ComfyUI 워크플로우 로드 및 수정
            with open(self.COMFYUI_WORKFLOW_PATH, "r") as f:
                workflow = json.load(f)

            # 워크플로우 파라미터 수정
            # 노드 76: 시작 이미지
            if "76" in workflow:
                # ComfyUI input 폴더로 이미지 복사 필요
                image_filename = os.path.basename(image_path)
                workflow["76"]["inputs"]["image"] = image_filename

            # 노드 75: 댄스 영상
            if "75" in workflow:
                video_filename = os.path.basename(video_path)
                workflow["75"]["inputs"]["video"] = video_filename
                workflow["75"]["inputs"]["frame_load_cap"] = num_frames

            # 노드 92: 텍스트 프롬프트
            if "92" in workflow:
                workflow["92"]["inputs"]["positive_prompt"] = video_prompt

            # ComfyUI input 폴더로 파일 복사
            comfyui_input_dir = os.path.expanduser("~/ComfyUI/input")
            if os.path.exists(comfyui_input_dir):
                import shutil
                shutil.copy(image_path, os.path.join(comfyui_input_dir, os.path.basename(image_path)))
                shutil.copy(video_path, os.path.join(comfyui_input_dir, os.path.basename(video_path)))

            # ComfyUI API로 워크플로우 실행
            prompt_url = f"{comfyui_url}/prompt"

            payload = {
                "prompt": workflow,
                "client_id": str(uuid.uuid4()),
            }

            async with aiohttp.ClientSession() as session:
                # 워크플로우 제출
                async with session.post(prompt_url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return NodeResult(success=False, error=f"ComfyUI API 오류: {error_text}")

                    result = await response.json()
                    prompt_id = result.get("prompt_id")

                if not prompt_id:
                    return NodeResult(success=False, error="prompt_id를 받지 못했습니다")

                print(f"[SteadyDancer] ComfyUI 작업 제출됨: {prompt_id}")

                # 완료 대기 (폴링)
                history_url = f"{comfyui_url}/history/{prompt_id}"
                max_wait = 1800  # 30분
                wait_interval = 10  # 10초
                waited = 0

                while waited < max_wait:
                    await asyncio.sleep(wait_interval)
                    waited += wait_interval

                    async with session.get(history_url) as response:
                        if response.status == 200:
                            history = await response.json()
                            if prompt_id in history:
                                outputs = history[prompt_id].get("outputs", {})
                                if outputs:
                                    print(f"[SteadyDancer] ComfyUI 작업 완료!")

                                    # 출력 파일 찾기 (노드 83이 비디오 출력)
                                    video_output = None
                                    for node_id, node_output in outputs.items():
                                        if "gifs" in node_output:
                                            video_output = node_output["gifs"][0]
                                            break
                                        if "videos" in node_output:
                                            video_output = node_output["videos"][0]
                                            break

                                    if video_output:
                                        output_filename = video_output.get("filename")
                                        output_subfolder = video_output.get("subfolder", "")
                                        output_type = video_output.get("type", "output")

                                        return NodeResult(
                                            success=True,
                                            data={
                                                "output_video": output_filename,
                                                "output_subfolder": output_subfolder,
                                                "output_type": output_type,
                                                "comfyui_prompt_id": prompt_id,
                                            }
                                        )

                                    return NodeResult(
                                        success=True,
                                        data={
                                            "comfyui_prompt_id": prompt_id,
                                            "comfyui_outputs": outputs,
                                        }
                                    )

                    print(f"[SteadyDancer] 대기 중... ({waited}s / {max_wait}s)")

                return NodeResult(success=False, error="ComfyUI 작업 타임아웃")

        except Exception as e:
            return NodeResult(success=False, error=str(e))

    async def _complete(self, state: WorkflowState) -> NodeResult:
        """워크플로우 완료"""
        data = state.get("data", {})

        return NodeResult(
            success=True,
            data={
                "workflow_completed": True,
                "summary": {
                    "generated_image": data.get("generated_image_path"),
                    "dance_video": data.get("dance_video_path"),
                    "video_title": data.get("video_title"),
                    "output_video": data.get("output_video"),
                    "comfyui_prompt_id": data.get("comfyui_prompt_id"),
                }
            }
        )

    async def _error_handler(self, state: WorkflowState) -> NodeResult:
        """에러 처리"""
        error = state.get("error", "Unknown error")
        retry_count = state.get("retry_count", 0)
        max_retries = 2

        print(f"[SteadyDancer] Error: {error}, retry={retry_count}")

        if retry_count < max_retries:
            failed_nodes = state.get("failed_nodes", [])
            retry_node = failed_nodes[-1] if failed_nodes else None

            if retry_node:
                print(f"[SteadyDancer] 재시도: {retry_node} ({retry_count + 1}/{max_retries})")
                return NodeResult(
                    success=True,
                    data={
                        "error": None,
                        "retry_count": retry_count + 1,
                    },
                    next_node=retry_node
                )

        return NodeResult(
            success=True,
            data={"error_handled": True, "original_error": error},
            next_node="complete"
        )
