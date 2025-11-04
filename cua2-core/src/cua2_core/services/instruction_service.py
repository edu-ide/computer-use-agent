import logging
import random
import time

from cua2_core.services.agent_utils.get_model import AVAILABLE_MODELS, get_model
from smolagents import ChatMessage, Model

logger = logging.getLogger(__name__)


class InstructionService:
    """Service for generating task instructions using LLM models"""

    available_models = AVAILABLE_MODELS
    seed_topics = [
        "web browsing",
        "email management",
        "calendar scheduling",
        "file management",
        "note-taking",
        "system settings",
        "text editing",
        "terminal commands",
    ]

    prompt_templates = [
        (
            "Generate a clear and specific task instruction for a desktop automation agent. "
            "The task should involve {topic} and be completable using a desktop computer. "
            "Do not assume any pre-existing files, emails, or resources exist on the system. "
            "Return only the task instruction, nothing else. Keep it simple and focused on a single action."
        ),
        (
            "Create a practical task instruction for desktop automation related to {topic}. "
            "The task should be straightforward and achievable in one application. "
            "Do not reference specific files or resources that may not exist locally. "
            "Provide only the task description without any additional explanation."
        ),
        (
            "Generate a specific {topic} task that a desktop automation agent can perform. "
            "The task should be concrete and not require multiple applications. "
            "Avoid assuming pre-existing documents, files, or local resources. "
            "Return just the task instruction."
        ),
        (
            "Provide a single, clear task instruction involving {topic} for a desktop agent. "
            "The task should be simple and focused. "
            "Do not assume any specific files or resources already exist on the computer. "
            "Output only the instruction."
        ),
        (
            "Think of a realistic {topic} task suitable for desktop automation. "
            "Keep it simple and achievable in one application. "
            "The task should not depend on pre-existing local files or resources. "
            "Return only the task."
        ),
    ]

    web_browsing_templates = [
        (
            "Generate a clear and specific web browsing task instruction for a desktop automation agent. "
            "The task should be goal-centric, focused on retrieving information or performing an action online. "
            "You can specify a URL or website to visit. "
            "Return only the task instruction, nothing else. Keep it simple and focused on a single goal."
        ),
        (
            "Create a practical web browsing task for desktop automation. "
            "The task should focus on finding specific information or completing an online action. "
            "Include a specific URL or website name if relevant to the goal. "
            "Provide only the task description without any additional explanation."
        ),
        (
            "Generate a specific web browsing task that a desktop automation agent can perform. "
            "The task should be about retrieving information or performing an action on a website. "
            "You may specify URLs or web addresses. Keep it concrete and single-purpose. "
            "Return just the task instruction."
        ),
        (
            "Provide a goal-oriented web browsing task instruction for a desktop agent. "
            "Focus on what information to find or what action to perform online. "
            "Specify a URL or website if it helps achieve the goal. "
            "Output only the instruction."
        ),
        (
            "Think of a realistic web browsing task suitable for desktop automation. "
            "The task should be about accessing online information or performing a web-based action. "
            "Include specific URLs or websites as needed. Keep it simple and goal-focused. "
            "Return only the task."
        ),
    ]

    default_prompt = (
        "Generate a clear and specific task instruction for a desktop automation agent. "
        "The task should be something that can be completed using a desktop computer, "
        "such as opening applications, browsing websites, or manipulating files. "
        "Do not assume any pre-existing files, emails, or resources exist on the system. "
        "Return only the task instruction, nothing else. the instruction must be not to complexe and not multi-app task. "
    )

    @staticmethod
    def get_random_prompt() -> str:
        """
        Generate a random prompt by selecting a random topic and template.
        Uses special templates for web browsing that allow URL specification.
        """
        random.seed(time.time_ns())

        topic = random.choice(InstructionService.seed_topics)

        if topic == "web browsing":
            template = random.choice(InstructionService.web_browsing_templates)
            return template

        template = random.choice(InstructionService.prompt_templates)
        return template.format(topic=topic)

    @staticmethod
    def generate_instruction(
        model_id: str, prompt: str | None = None, use_random: bool = True
    ) -> str:
        """
        Generate a task instruction using the specified model

        Args:
            model_id: The ID of the model to use
            prompt: Optional custom prompt. If None, uses default or random prompt
            use_random: If True, uses random prompts for variety. If False, uses default prompt
        """

        if model_id not in InstructionService.available_models:
            available_models_str = ", ".join(InstructionService.available_models)
            raise ValueError(
                f"Invalid model_id '{model_id}'. Must be one of: {available_models_str}"
            )

        try:
            logger.info(f"Generating instruction with model: {model_id}")

            model: Model = get_model(model_id)

            if prompt:
                generation_prompt = prompt
            elif use_random:
                generation_prompt = InstructionService.get_random_prompt()
            else:
                generation_prompt = InstructionService.default_prompt

            instruction = model([ChatMessage(role="user", content=generation_prompt)])
            logger.info(
                f"Successfully generated instruction with {model_id}: {instruction.content[:100]}..."
            )
            return instruction.content

        except Exception as e:
            logger.error(f"Error generating instruction with {model_id}: {str(e)}")
            raise Exception(f"Failed to generate instruction: {str(e)}")

    @staticmethod
    def get_available_models() -> list[str]:
        """Get the list of available model IDs"""
        return InstructionService.available_models

    @staticmethod
    def get_random_topic() -> str:
        """Get a random topic from the seed topics"""
        return random.choice(InstructionService.seed_topics)


if __name__ == "__main__":
    instruction = InstructionService.generate_instruction(
        model_id="Qwen/Qwen3-VL-8B-Instruct"
    )
    print(instruction)
