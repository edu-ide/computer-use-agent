import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class InstructionService:
    """Service for loading task instructions from pregenerated pool"""

    # Path to the pregenerated instructions file
    INSTRUCTIONS_FILE_PATH = (
        Path(__file__).parent
        / "agent_utils"
        / "instruction_utils"
        / "pregenerated_instructions.json"
    )
    # Cache for loaded instructions
    _pregenerated_instructions: list[str] | None = None

    @staticmethod
    def _load_pregenerated_instructions() -> list[str]:
        """
        Load pregenerated instructions from the JSON file.
        Uses lazy loading and caching to avoid repeated file reads.

        Returns:
            List of pregenerated instruction strings

        Raises:
            FileNotFoundError: If the pregenerated instructions file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        # Return cached instructions if already loaded
        if InstructionService._pregenerated_instructions is not None:
            return InstructionService._pregenerated_instructions

        file_path = InstructionService.INSTRUCTIONS_FILE_PATH

        if not file_path.exists():
            raise FileNotFoundError(
                f"Pregenerated instructions file not found at {file_path}. "
                f"Please ensure the file exists."
            )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                instructions = json.load(f)

            if not isinstance(instructions, list):
                raise ValueError(
                    f"Invalid format in pregenerated instructions file. Expected a list, got {type(instructions)}"
                )

            # Cache the instructions
            InstructionService._pregenerated_instructions = instructions
            logger.info(
                f"Loaded {len(instructions)} pregenerated instructions from {file_path}"
            )
            return instructions

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {file_path}: {str(e)}")
            raise

    @staticmethod
    def get_random_instruction() -> str:
        """
        Get a random instruction from the pregenerated pool.

        Returns:
            A random instruction string

        Raises:
            FileNotFoundError: If the pregenerated instructions file doesn't exist
            IndexError: If the pregenerated instructions list is empty
        """
        instructions = InstructionService._load_pregenerated_instructions()

        if not instructions:
            raise IndexError("Pregenerated instructions list is empty")

        return random.choice(instructions)


if __name__ == "__main__":
    instruction = InstructionService.get_random_instruction()
    print(instruction)
