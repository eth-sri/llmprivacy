from src.configs import ModelConfig
from src.models import BaseModel
from src.prompts import Prompt
from src.utils.limiter import RateLimiter
from typing import List, Tuple, Iterator
from tqdm import tqdm
import time

from vertexai.preview.language_models import ChatModel, TextGenerationModel


# https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/text-bison?project=bison-397406


class GCPPaLM(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

        if "temperature" not in self.config.args.keys():
            self.config.args["temperature"] = 0.0
        if "max_tokens" in self.config.args.keys():
            self.config.args["max_output_tokens"] = self.config.args["max_tokens"]
            del self.config.args["max_tokens"]

        if "max_output_tokens" not in self.config.args.keys():
            self.config.args["max_output_tokens"] = 600

        if "chat" in config.name:
            self.model = ChatModel.from_pretrained(config.name)
        else:
            self.model = TextGenerationModel.from_pretrained(config.name)
        assert self.model is not None

    def predict(self, input: Prompt, **kwargs) -> str:
        # parameters = {
        #     "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        #     "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
        #     "top_p": 0.95,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        #     "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
        # }

        if "chat" in self.config.name:
            chat = self.model.start_chat(  # type: ignore
                context=input.system_prompt,
            )

            response = chat.send_message(  # type: ignore
                self.apply_model_template(input.get_prompt()), **self.config.args
            )
        else:
            response = self.model.predict(  # type: ignore
                self.apply_model_template(input.get_prompt()), **self.config.args
            )

        return response.text

    def predict_string(self, input: str, **kwargs) -> str:
        response = self.model.predict(input, **self.config.args)  # type: ignore

        return response.text

    def predict_multi(
        self, inputs: List[Prompt], **kwargs
    ) -> Iterator[Tuple[Prompt, str]]:
        rl = RateLimiter(9, 30)

        ids_to_do = list(range(len(inputs)))

        while len(ids_to_do) > 0:
            for id in tqdm(ids_to_do):
                if not rl.record():
                    print("Rate limit exceeded, sleeping for 10 seconds")
                    time.sleep(10)
                    continue

                orig = inputs[id]
                answer = self.predict(inputs[id])

                yield (orig, answer)
                ids_to_do.remove(id)
