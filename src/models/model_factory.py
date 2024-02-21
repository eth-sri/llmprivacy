from src.configs import ModelConfig

from .chain_model import ChainModel
from .model import BaseModel
from .multi_model import MultiModel
from .open_ai import OpenAIGPT
from .hf_model import HFModel
from .gcp.gcp_model import GCPPaLM
from .poe.poe_model import PoeModel


def get_model(config: ModelConfig) -> BaseModel:
    if config.provider == "openai" or config.provider == "azure":
        return OpenAIGPT(config)
    elif config.provider == "hf":
        return HFModel(config)
    elif config.provider == "gcp":
        return GCPPaLM(config)
    elif config.provider == "poe":
        return PoeModel(config)
    elif config.provider == "loc":
        if config.name == "multi":
            models = []
            for sub_cfg in config.submodels:
                models.append(get_model(sub_cfg))

            return MultiModel(config, models)
        if config.name == "chain":
            models = []
            for sub_cfg in config.submodels:
                models.append(get_model(sub_cfg))

            return ChainModel(config, models)
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
