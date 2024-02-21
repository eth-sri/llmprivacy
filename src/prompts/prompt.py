from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.anonymization import anonymize_presidio


@dataclass
class Prompt:
    system_prompt: Optional[str] = ""  # The system prompt
    role: Optional[str] = ""  # The role of the prompt
    header: str = ""  # Initial header
    intermediate: str = ""  # Intermediate text
    footer: str = ""  # Final footer
    target: str = ""  # Name of the target attribute
    original_point: Dict[str, Any] = field(default_factory=dict)  # Original point
    gt: Optional[str] = None  # Ground truth
    answer: Optional[str] = None  # Answer given to the prompt
    shots: List[str] = field(default_factory=list)
    id: int = -1
    template: str = "{header}\n{shots}\n{intermediate}\n\n{footer}\n\n{answer}"

    def get_prompt(self, show_answer=False):
        if show_answer:
            return self.template.format(
                header=self.header,
                shots="\n\n".join(self.shots),
                intermediate=self.intermediate,
                footer=self.footer,
                answer=self.gt,
            )
        else:
            return self.template.format(
                header=self.header,
                shots="\n\n".join(self.shots),
                intermediate=self.intermediate,
                footer=self.footer,
                answer="",
            )

    def anonymize(self, anonymizer: str = "presidio"):
        if anonymizer == "presidio":
            self.intermediate = anonymize_presidio(self.intermediate)
            self.shots = [anonymize_presidio(s) for s in self.shots]
        else:
            print("Anonymizer not supported")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, dict) -> "Prompt":
        return cls(**dict)

    def get_copy(self):
        return Prompt(
            header=self.header,
            intermediate=self.intermediate,
            footer=self.footer,
            target=self.target,
            original_point=self.original_point,
            gt=self.gt,
            answer=self.answer,
            shots=self.shots,
            id=self.id,
            template=self.template,
        )

    def __str__(self) -> str:

        sys_prompt = self.system_prompt if self.system_prompt else "No system prompt"

        return f"Prompt: {self.id}\n=============\nSystem Prompt\n=============\n{sys_prompt}\n=============\nPrompt\n=============\n{self.get_prompt()}"


@dataclass
class Conversation:
    system_prompt: str
    prompts: List[Prompt]

    def __init__(self, system_prompt: str, prompts: List[Prompt]) -> None:
        self.system_prompt = system_prompt
        for prompt in prompts:
            prompt.system_prompt = ""
            prompt.template = "{intermediate}"
        self.prompts = prompts

    def set_system_prompt(self, system_prompt: str) -> "Conversation":
        self.system_prompt = system_prompt
        return self

    def get_copy(self):
        return Conversation(
            system_prompt=self.system_prompt,
            prompts=self.prompts
        )

    def swap_roles(self, swap_dict: Dict[str, str]):
        for prompt in self.prompts:
            if prompt.role in swap_dict:
                prompt.role = swap_dict[prompt.role]
            else:
                print("Role not found!")

    def add_prompt(self, prompt: Prompt):
        self.prompts.append(prompt)
