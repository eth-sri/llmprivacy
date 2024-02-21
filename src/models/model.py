from abc import ABC, abstractmethod
from typing import List, Tuple, Iterator

from src.configs import ModelConfig
from src.prompts import Prompt, Conversation


class BaseModel(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None

    @abstractmethod
    def predict(self, input: Prompt, **kwargs) -> str:
        pass

    @abstractmethod
    def predict_multi(
        self, inputs: List[Prompt], **kwargs
    ) -> Iterator[Tuple[Prompt, str]]:
        pass

    @abstractmethod
    def predict_string(self, input: str, **kwargs) -> str:
        pass

    def continue_conversation(self, input: Conversation, **kwargs) -> str:
        raise NotImplementedError

    def apply_model_template(self, input: str, **kwargs) -> str:
        return self.config.model_template.format(prompt=input)
