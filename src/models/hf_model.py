from typing import List, Dict, Iterator, Tuple
import torch
from src.configs import ModelConfig
from src.prompts import Prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model import BaseModel


class HFModel(BaseModel):
    curr_models: Dict[str, AutoModelForCausalLM] = {}

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if (
            config.name not in self.curr_models
        ):  # Hacky but avoids loading the same model multiple times
            self.curr_models[config.name] = AutoModelForCausalLM.from_pretrained(
                config.name,
                torch_dtype=(
                    torch.float16 if config.dtype == "float16" else torch.float32
                ),
                device_map=config.device,
            )
        self.model: AutoModelForCausalLM = self.curr_models[config.name]
        # self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        #     config.name,
        #     torch_dtype=torch.float16 if config.dtype == "float16" else torch.float32,
        #     device_map=config.device,
        # )
        self.device = self.model.device
        if config.tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

    def predict(self, input: Prompt, **kwargs):
        text = input.get_prompt().rstrip()

        model_text = self.apply_model_template(text)

        input_ids = self.tokenizer.encode(
            model_text,
            return_tensors="pt",
        ).to(self.device)
        input_length = len(input_ids[0])

        output = self.model.generate(input_ids, **self.config.args)

        # For decoder only models:
        out_ids = output[:, input_length:]

        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

    def predict_multi(
        self, inputs: List[Prompt], **kwargs
    ) -> Iterator[Tuple[Prompt, str]]:
        new_inputs = []
        for input in inputs:
            text = input.get_prompt().rstrip()
            new_inputs.append(self.apply_model_template(text))

        if "max_workers" in kwargs:
            batch_size = kwargs["max_workers"]
        else:
            batch_size = 1

        for i in range(0, len(new_inputs), batch_size):
            end = min(i + batch_size, len(new_inputs))
            new_inputs_batch = new_inputs[i:end]

            model_inputs = self.tokenizer(
                new_inputs_batch,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).to(self.device)
            input_length = len(model_inputs["input_ids"][0])

            output = self.model.generate(**model_inputs, **self.config.args)

            outs_str = self.tokenizer.batch_decode(
                output[:, input_length:], skip_special_tokens=True
            )
            for j in range(len(outs_str)):
                yield (inputs[i + j], outs_str[j])
            # yield (inputs[i], outs_str[0])

            # results.extend(outs_str)

        # return results

    def predict_string(self, input: str, **kwargs) -> str:
        return self.predict(Prompt(intermediate=input), **kwargs)
