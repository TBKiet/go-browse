from dataclasses import dataclass
from ..trajectory_data import StepData, TrajectoryData

def flatten_messages(messages: list[dict]):
    flattened_messages = []
    for message in messages:
        role = message['role']
        text = '\n\n'.join(c['text'] for c in message['content'])
        flattened_messages.append({'role': role, 'content': text})
    return flattened_messages


class BasePromptBuilder:
    
    def build_trajectory_messages(self, trajectory_data: TrajectoryData, char_limit: int = -1) -> list[dict]: 
        raise NotImplementedError
    
    def build_messages(self, goal: str, current_step: StepData, history: list[StepData], char_limit: int = -1) -> dict:
        raise NotImplementedError
    
    def pretty_prompt_string(self, messages: list[dict]) -> str:
        """
        Convert a list of prompt messages into a single string suitable for printing.

        Args:
            messages (list[dict]): The list of messages to convert.

        Returns:
            str: The processed stringified prompt suitable for printing.
        """
        prompt_text_strings = []
        for message in messages:
            prompt_text_strings.append(message["content"])
        full_prompt_txt = "\n".join(prompt_text_strings)
        return full_prompt_txt        

PROMPT_BUILDER_REGISTRY: dict[str, BasePromptBuilder] = {}

class PromptBuilderFactory:

    def create_prompt_builder(self, name: str):
        if name not in PROMPT_BUILDER_REGISTRY:
            raise ValueError(f"Unknown prompt builder: {name}")
        return PROMPT_BUILDER_REGISTRY[name]()
    
    @staticmethod
    def register(cls, aliases: str | tuple[str] = tuple()):
        PROMPT_BUILDER_REGISTRY[cls.__name__] = cls

        if isinstance(aliases, str):
            aliases = (aliases,)

        for name in aliases:
            PROMPT_BUILDER_REGISTRY[name] = cls
