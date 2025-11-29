"""
유튜브 콘텐츠 자동화 워크플로우 - LangGraph 기반

워크플로우 단계:
1. 채널 분석 & 아이디어 수집
2. 콘텐츠 기획 & 스토리보드 작성
3. 자막/스크립트 제작
4. 영상 업로드
"""

import asyncio
from typing import Any, Dict, List, Optional

from .workflow_base import (
    WorkflowBase,
    WorkflowConfig,
    WorkflowNode,
    WorkflowState,
    NodeResult,
    NodeStatus,
)


class YouTubeContentWorkflow(WorkflowBase):
    """
    유튜브 콘텐츠 자동화 워크플로우

    Flow:
    1. analyze_trends - 트렌드 분석 및 아이디어 수집
    2. research_topic - 주제 리서치 및 자료 수집
    3. generate_ideas - 콘텐츠 아이디어 도출
    4. create_storyboard - 스토리보드 작성
    5. write_script - 스크립트/대본 작성
    6. generate_subtitles - 자막 생성
    7. prepare_upload - 업로드 준비 (썸네일, 제목, 설명)
    8. upload_video - 유튜브 업로드
    9. complete - 완료
    """

    def __init__(self, agent_runner=None):
        """
        Args:
            agent_runner: VLM 에이전트 실행기
        """
        super().__init__()
        self._agent_runner = agent_runner

    @property
    def config(self) -> WorkflowConfig:
        return WorkflowConfig(
            id="youtube-content",
            name="유튜브 콘텐츠 자동화",
            description="채널 분석부터 영상 업로드까지 유튜브 콘텐츠 제작 전 과정을 자동화합니다.",
            icon="YouTube",
            color="#ff0000",
            category="content",
            parameters=[
                {
                    "name": "channel_url",
                    "type": "string",
                    "label": "채널 URL",
                    "placeholder": "https://youtube.com/@channel",
                    "required": False,
                },
                {
                    "name": "topic",
                    "type": "string",
                    "label": "콘텐츠 주제/키워드",
                    "placeholder": "예: AI 자동화, 프로그래밍 튜토리얼",
                    "required": True,
                },
                {
                    "name": "video_style",
                    "type": "string",
                    "label": "영상 스타일",
                    "placeholder": "예: 튜토리얼, 브이로그, 리뷰",
                    "default": "튜토리얼",
                },
                {
                    "name": "target_length",
                    "type": "number",
                    "label": "목표 영상 길이 (분)",
                    "default": 10,
                    "min": 1,
                    "max": 60,
                },
                {
                    "name": "auto_upload",
                    "type": "boolean",
                    "label": "자동 업로드",
                    "default": False,
                },
            ],
        )

    @property
    def nodes(self) -> List[WorkflowNode]:
        return [
            WorkflowNode(
                name="analyze_trends",
                display_name="트렌드 분석",
                description="유튜브 트렌드 분석 및 인기 콘텐츠 파악",
                on_success="research_topic",
                on_failure="error_handler",
            ),
            WorkflowNode(
                name="research_topic",
                display_name="주제 리서치",
                description="주제 관련 자료 및 레퍼런스 수집",
                on_success="generate_ideas",
                on_failure="error_handler",
            ),
            WorkflowNode(
                name="generate_ideas",
                display_name="아이디어 도출",
                description="콘텐츠 아이디어 도출 및 선정",
                on_success="create_storyboard",
                on_failure="error_handler",
            ),
            WorkflowNode(
                name="create_storyboard",
                display_name="스토리보드",
                description="영상 스토리보드 및 구성안 작성",
                on_success="write_script",
                on_failure="error_handler",
            ),
            WorkflowNode(
                name="write_script",
                display_name="스크립트 작성",
                description="스크립트/대본 작성",
                on_success="generate_subtitles",
                on_failure="error_handler",
            ),
            WorkflowNode(
                name="generate_subtitles",
                display_name="자막 생성",
                description="자막 파일 생성 (SRT/VTT)",
                on_success="prepare_upload",
                on_failure="error_handler",
            ),
            WorkflowNode(
                name="prepare_upload",
                display_name="업로드 준비",
                description="썸네일, 제목, 설명, 태그 준비",
                on_success="check_auto_upload",
                on_failure="error_handler",
            ),
            WorkflowNode(
                name="check_auto_upload",
                display_name="업로드 확인",
                description="자동 업로드 여부 확인",
                on_success="upload_video",
                on_failure="complete",  # 자동 업로드 비활성화 시 완료
            ),
            WorkflowNode(
                name="upload_video",
                display_name="영상 업로드",
                description="유튜브 스튜디오에서 영상 업로드",
                on_success="complete",
                on_failure="error_handler",
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
        return "analyze_trends"

    async def execute_node(self, node_name: str, state: WorkflowState) -> NodeResult:
        """노드별 실행 로직"""
        handlers = {
            "analyze_trends": self._analyze_trends,
            "research_topic": self._research_topic,
            "generate_ideas": self._generate_ideas,
            "create_storyboard": self._create_storyboard,
            "write_script": self._write_script,
            "generate_subtitles": self._generate_subtitles,
            "prepare_upload": self._prepare_upload,
            "check_auto_upload": self._check_auto_upload,
            "upload_video": self._upload_video,
            "complete": self._complete,
            "error_handler": self._error_handler,
        }

        handler = handlers.get(node_name)
        if handler:
            return await handler(state)

        return NodeResult(success=False, error=f"Unknown node: {node_name}")

    async def _analyze_trends(self, state: WorkflowState) -> NodeResult:
        """트렌드 분석 및 아이디어 수집"""
        try:
            parameters = state.get("parameters", {})
            topic = parameters.get("topic", "")
            channel_url = parameters.get("channel_url", "")

            if self._agent_runner:
                instruction = f"""
                Analyze YouTube trends for the topic "{topic}":

                1. Open YouTube (https://youtube.com)
                2. Search for "{topic}" related videos
                3. Analyze top performing videos:
                   - View counts
                   - Upload dates
                   - Video lengths
                   - Thumbnail styles
                   - Title patterns
                4. {"Also analyze the channel: " + channel_url if channel_url else ""}
                5. Identify trending formats and styles
                6. Note down successful content patterns

                Report findings about what makes content successful in this niche.
                """
                result = await self._agent_runner.run_instruction(instruction)
                if not result.success:
                    return NodeResult(success=False, error=result.error)

            return NodeResult(
                success=True,
                data={
                    "trends_analyzed": True,
                    "topic": topic,
                }
            )
        except Exception as e:
            return NodeResult(success=False, error=str(e))

    async def _research_topic(self, state: WorkflowState) -> NodeResult:
        """주제 리서치 및 자료 수집"""
        try:
            parameters = state.get("parameters", {})
            topic = parameters.get("topic", "")

            if self._agent_runner:
                instruction = f"""
                Research the topic "{topic}" thoroughly:

                1. Search Google for "{topic}" latest information
                2. Find authoritative sources and references
                3. Collect key facts, statistics, and insights
                4. Look for unique angles or perspectives
                5. Note any controversies or debates
                6. Find relevant examples and case studies

                Compile research notes for content creation.
                """
                result = await self._agent_runner.run_instruction(instruction)
                if not result.success:
                    return NodeResult(success=False, error=result.error)

            return NodeResult(
                success=True,
                data={"research_completed": True}
            )
        except Exception as e:
            return NodeResult(success=False, error=str(e))

    async def _generate_ideas(self, state: WorkflowState) -> NodeResult:
        """콘텐츠 아이디어 도출"""
        try:
            parameters = state.get("parameters", {})
            topic = parameters.get("topic", "")
            video_style = parameters.get("video_style", "튜토리얼")

            if self._agent_runner:
                instruction = f"""
                Generate content ideas for "{topic}" in {video_style} style:

                1. Based on trend analysis and research, create 5 video ideas
                2. For each idea, provide:
                   - Catchy title (Korean)
                   - Hook/opening concept
                   - Main value proposition
                   - Target audience
                3. Select the best idea based on:
                   - Uniqueness
                   - Search potential
                   - Engagement potential
                4. Create a brief content outline

                Present the selected idea with reasoning.
                """
                result = await self._agent_runner.run_instruction(instruction)
                if not result.success:
                    return NodeResult(success=False, error=result.error)

            return NodeResult(
                success=True,
                data={
                    "ideas_generated": True,
                    "selected_idea": topic,
                }
            )
        except Exception as e:
            return NodeResult(success=False, error=str(e))

    async def _create_storyboard(self, state: WorkflowState) -> NodeResult:
        """스토리보드 작성"""
        try:
            parameters = state.get("parameters", {})
            target_length = parameters.get("target_length", 10)
            video_style = parameters.get("video_style", "튜토리얼")

            if self._agent_runner:
                instruction = f"""
                Create a detailed storyboard for a {target_length}-minute {video_style} video:

                1. Structure the video into clear sections:
                   - Hook (0:00-0:30): Attention grabber
                   - Intro (0:30-1:00): Topic introduction
                   - Main content (1:00-{target_length-2}:00): Core information
                   - Summary ({target_length-2}:00-{target_length-1}:00): Key takeaways
                   - CTA ({target_length-1}:00-{target_length}:00): Call to action

                2. For each section, note:
                   - Visual elements needed
                   - Key points to cover
                   - Transitions
                   - B-roll suggestions

                3. Plan engagement elements:
                   - Questions to ask viewers
                   - Interactive moments
                   - Subscribe/like reminders

                Create a structured storyboard document.
                """
                result = await self._agent_runner.run_instruction(instruction)
                if not result.success:
                    return NodeResult(success=False, error=result.error)

            return NodeResult(
                success=True,
                data={
                    "storyboard_created": True,
                    "video_length": target_length,
                }
            )
        except Exception as e:
            return NodeResult(success=False, error=str(e))

    async def _write_script(self, state: WorkflowState) -> NodeResult:
        """스크립트/대본 작성"""
        try:
            parameters = state.get("parameters", {})
            target_length = parameters.get("target_length", 10)

            if self._agent_runner:
                instruction = f"""
                Write a complete video script based on the storyboard:

                1. Write natural, conversational Korean script
                2. Include:
                   - Exact dialogue/narration
                   - [VISUAL] cues for what to show
                   - [MUSIC] cues for background music
                   - [SFX] for sound effects
                   - [TRANSITION] markers

                3. Script should be approximately {target_length * 150} words
                   (about 150 words per minute)

                4. Include timestamps for each section

                5. Add emphasis markers for important points

                Save the script in a readable format.
                """
                result = await self._agent_runner.run_instruction(instruction)
                if not result.success:
                    return NodeResult(success=False, error=result.error)

            return NodeResult(
                success=True,
                data={"script_written": True}
            )
        except Exception as e:
            return NodeResult(success=False, error=str(e))

    async def _generate_subtitles(self, state: WorkflowState) -> NodeResult:
        """자막 생성"""
        try:
            if self._agent_runner:
                instruction = """
                Generate subtitle files from the script:

                1. Create SRT format subtitles:
                   - Proper timing (2-3 seconds per subtitle)
                   - Max 2 lines per subtitle
                   - Max 42 characters per line

                2. Format example:
                   1
                   00:00:00,000 --> 00:00:02,500
                   안녕하세요, 오늘은

                   2
                   00:00:02,500 --> 00:00:05,000
                   AI 자동화에 대해 알아보겠습니다

                3. Also create:
                   - Korean subtitles (main)
                   - English subtitles (translation)

                Save as .srt files.
                """
                result = await self._agent_runner.run_instruction(instruction)
                if not result.success:
                    return NodeResult(success=False, error=result.error)

            return NodeResult(
                success=True,
                data={
                    "subtitles_generated": True,
                    "subtitle_formats": ["srt", "vtt"],
                }
            )
        except Exception as e:
            return NodeResult(success=False, error=str(e))

    async def _prepare_upload(self, state: WorkflowState) -> NodeResult:
        """업로드 준비"""
        try:
            parameters = state.get("parameters", {})
            topic = parameters.get("topic", "")

            if self._agent_runner:
                instruction = f"""
                Prepare all assets for YouTube upload:

                1. Create video title options (Korean):
                   - Main title with keyword "{topic}"
                   - Include hook or benefit
                   - Under 60 characters

                2. Write video description:
                   - Summary in first 2 lines (shown in search)
                   - Timestamps for each section
                   - Links and resources
                   - Social media links
                   - Hashtags

                3. Generate tags:
                   - Primary keyword: {topic}
                   - Related keywords (10-15 tags)
                   - Mix of broad and specific

                4. Thumbnail concept:
                   - Text overlay suggestion
                   - Color scheme
                   - Facial expression if applicable

                5. Select category and settings:
                   - Video category
                   - Language
                   - License
                   - Comments settings

                Compile all metadata for upload.
                """
                result = await self._agent_runner.run_instruction(instruction)
                if not result.success:
                    return NodeResult(success=False, error=result.error)

            return NodeResult(
                success=True,
                data={
                    "upload_prepared": True,
                    "metadata_ready": True,
                }
            )
        except Exception as e:
            return NodeResult(success=False, error=str(e))

    async def _check_auto_upload(self, state: WorkflowState) -> NodeResult:
        """자동 업로드 여부 확인"""
        parameters = state.get("parameters", {})
        auto_upload = parameters.get("auto_upload", False)

        if auto_upload:
            return NodeResult(
                success=True,  # on_success -> upload_video
                data={"auto_upload_enabled": True}
            )
        else:
            return NodeResult(
                success=False,  # on_failure -> complete
                data={
                    "auto_upload_enabled": False,
                    "message": "자동 업로드 비활성화. 수동 업로드 필요."
                }
            )

    async def _upload_video(self, state: WorkflowState) -> NodeResult:
        """유튜브 업로드"""
        try:
            if self._agent_runner:
                instruction = """
                Upload video to YouTube Studio:

                1. Open YouTube Studio (https://studio.youtube.com)
                2. Click "CREATE" -> "Upload videos"
                3. Select the video file
                4. Fill in metadata:
                   - Title
                   - Description
                   - Thumbnail
                   - Playlist
                   - Tags
                5. Set visibility:
                   - Public / Unlisted / Private / Scheduled
                6. Add end screens and cards
                7. Upload subtitles
                8. Review and publish

                Confirm successful upload and get video URL.
                """
                result = await self._agent_runner.run_instruction(instruction)
                if not result.success:
                    return NodeResult(success=False, error=result.error)

            return NodeResult(
                success=True,
                data={
                    "video_uploaded": True,
                    "upload_status": "completed",
                }
            )
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
                    "trends_analyzed": data.get("trends_analyzed", False),
                    "research_completed": data.get("research_completed", False),
                    "ideas_generated": data.get("ideas_generated", False),
                    "storyboard_created": data.get("storyboard_created", False),
                    "script_written": data.get("script_written", False),
                    "subtitles_generated": data.get("subtitles_generated", False),
                    "upload_prepared": data.get("upload_prepared", False),
                    "video_uploaded": data.get("video_uploaded", False),
                }
            }
        )

    async def _error_handler(self, state: WorkflowState) -> NodeResult:
        """에러 처리"""
        error = state.get("error", "Unknown error")
        print(f"[YouTubeWorkflow] Error: {error}")

        return NodeResult(
            success=False,
            error=error,
            data={"error_handled": True}
        )
