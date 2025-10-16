from typing import Annotated, TypeAlias

from pydantic import Field
from smolagents import Model

from backend.models.models import AgentType
from backend.services.agents.normalized_1000_agent import Normalized1000Agent
from backend.services.agents.normalized_agent import NormalizedAgent
from backend.services.agents.pixel_coordonates_agent import PixelCoordinatesAgent
from backend.services.agents.prompt import (
    Normalized1000CoordinatesSystemPrompt,
    NormalizedCoordinatesSystemPrompt,
    PixelCoordinatesSystemPrompt,
)
from computer_use_studio import Sandbox

Agent: TypeAlias = Annotated[
    PixelCoordinatesAgent | Normalized1000Agent | NormalizedAgent,
    Field(discriminator="AGENT_TYPE"),
]


def get_agent(
    model: Model,
    desktop: Sandbox,
    agent_type: AgentType,
    prompt_type: str,
    data_dir: str,
    **kwargs,
) -> Agent:
    """Get the agent by type"""
    if agent_type == AgentType.PIXEL_COORDINATES:
        return PixelCoordinatesAgent(
            model=model,
            desktop=desktop,
            system_prompt=PixelCoordinatesSystemPrompt[prompt_type].value,
            data_dir=data_dir,
            **kwargs,
        )
    elif agent_type == AgentType.NORMALIZED_1000_COORDINATES:
        return Normalized1000Agent(
            model=model,
            desktop=desktop,
            system_prompt=Normalized1000CoordinatesSystemPrompt[prompt_type].value,
            data_dir=data_dir,
            **kwargs,
        )
    elif agent_type == AgentType.NORMALIZED_COORDINATES:
        return Normalized1000Agent(
            model=model,
            desktop=desktop,
            system_prompt=NormalizedCoordinatesSystemPrompt[prompt_type].value,
            data_dir=data_dir,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid agent type: {agent_type}")
