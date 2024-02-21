from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel as PBM
from pydantic import Extra, Field

from src.utils.string_utils import string_hash


class Task(Enum):
    ACS = "ACS"  # American Community Survey
    PAN = "PAN"  # PAN 2018
    REDDIT = "REDDIT"  # Reddit
    # Synthetic options
    CHAT = "CHAT"  # Multiturn conversations
    CHAT_EVAL = "CHAT_EVAL"  # Evaluation of multiturn conversations
    SYNTHETIC = "SYNTHETIC"  # Synthetic Reddit comments generation


class ModelConfig(PBM):
    name: str = Field(description="Name of the model")
    tokenizer_name: Optional[str] = Field(
        None, description="Name of the tokenizer to use"
    )
    provider: str = Field(description="Provider of the model")
    dtype: str = Field(
        "float16", description="Data type of the model (only used for local models)"
    )
    device: str = Field(
        "auto", description="Device to use for the model (only used for local models)"
    )
    max_workers: int = Field(
        1, description="Number of workers (Batch-size) to use for parallel generation"
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the model upon generation",
    )
    model_template: str = Field(
        default="{prompt}",
        description="Template to use for the model (only used for local models)",
    )
    prompt_template: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the prompt"
    )
    submodels: List["ModelConfig"] = Field(
        default_factory=list, description="Submodels to use"
    )
    multi_selector: str = Field(
        default="majority", description="How to select the final answer"
    )

    def get_name(self) -> str:
        if self.name == "multi":
            return "multi" + "_".join(
                [submodel.get_name() for submodel in self.submodels]
            )
        if self.name == "chain":
            return "chain_" + "_".join(
                [submodel.get_name() for submodel in self.submodels]
            )
        if self.provider == "hf":
            return self.name.split("/")[-1]
        else:
            return self.name


class BasePromptConfig(PBM):
    # Contains prompt attributes which pertain to the header and footer of the prompt
    # These attributes are not used in the intermediate text (i.e. the meat of the prompt)
    modifies: List[str] = Field(
        default_factory=list,
        description="Whether this prompt config is used to modify existing prompts",
    )
    num_answers: int = Field(3, description="Number of answer given by the model")
    num_shots: int = Field(
        0, description="Number of shots to be presented to the model"
    )
    cot: bool = Field(False, description="Whether to use COT prompting")
    use_qa: bool = Field(
        False, description="Whether to present answer options to the model"
    )
    header: Optional[str] = Field(
        default=None,
        description="In case we want to set a specific header for the prompt.",
    )
    footer: Optional[str] = Field(
        default=None,
        description="In case we want to set a specific footer for the prompt.",
    )

    # Workaround to use this as a pure modifier as well
    def __init__(self, **data):
        super().__init__(**data)
        self.modifies = list(data.keys())

    def get_filename(self) -> str:
        file_path = ""
        for attr in vars(self):
            if attr in ["dryrun", "save_prompts", "header", "footer", "modifies"]:
                continue

            if "_" in attr:
                attr_short = attr.split("_")[1][:4]
            else:
                attr_short = attr[:4]

            file_path += f"{attr_short}={getattr(self, attr)}_"
        return file_path[:-1] + ".txt"


class ACSConfig(PBM):
    # ACS specific attributes
    state: str = Field(description="State", default="NY")
    given_attrs: List[str] = Field(
        default_factory=list, description="Attributes to be given to the model"
    )
    target: str = Field(..., description="Attributes to be predicted by the model")
    total_queries: int = Field(
        ..., description="Total number of queries to be presented to the model"
    )
    # Prompt Loading
    prompt_path: str = Field(
        default=None, description="Path to a file already containing prompts"
    )
    # Specifies how header and footer of the text
    outer_prompt: BasePromptConfig = Field(
        default=None,
        description="Config to be used for the outer prompt i.e. header and footer",
    )
    # These specify how to create the intermediate text of the prompt
    inner_template: str = Field(
        default="{orig}", description="Template to be used for the inner prompt"
    )
    level: Optional[int] = Field(
        default=None,
        ge=0,
        le=4,
        description="How much information should be hidden on a scale 0-4 (inclusive)",
    )
    creativity: float = Field(
        0.0,
        description="Creativity parameter for prompt creation -> 0 gives the base prompts while 1 always varies the original prompt",
    )
    complexity: float = Field(
        0.0,
        description="Complexity parameter for prompt creation -> 0 creates easier prompts while 1 creates harder prompts",
    )
    lm_finetune: bool = Field(
        False,
        description="Whether we make a more coherent prompt using a weaker language model (GPT3.5)",
    )
    validator: str = Field(
        default="default",
        description="Validator to be used to check if an answer is valid",
    )

    # Cannot be used as spacy requires a pydantic version that does not have this feature yet
    # @model_validator(mode="after")
    # def check_mapping_config(self):
    #     if self.prompt_path is not None and self.outer_prompt is not None:
    #         raise ValidationError(
    #             "Cannot have both outer_prompt and prompt_loading set"
    #         )
    #     elif self.modifier is not None and self.prompt_path is None:
    #         raise ValidationError("Cannot have modifier without prompt_path set")
    #     return self

    class Config:
        extra = Extra.forbid

    # Some space saving for windows
    def get_filename(self) -> str:
        file_path = f"acs_{self.state}_"
        for attr in vars(self):
            if attr == "prompt_path":
                if self.prompt_path is not None:
                    file_path += "prom=loaded_"
                continue

            if attr == "outer_prompt":
                file_path += self.outer_prompt.get_filename() + "_"
                continue

            if attr == "inner_template":
                curr = "temp=" + getattr(self, attr)
                if self.outer_prompt.header is not None:
                    curr += "header=" + self.outer_prompt.header
                if self.outer_prompt.footer is not None:
                    curr += "footer=" + self.outer_prompt.footer

                curr = string_hash(curr)

                file_path += f"temp={curr}_"
                continue

            if attr == "given_attrs":
                file_path += "[" + ",".join(self.given_attrs) + "]_"
                continue

            if "_" in attr:
                attr_short = attr.split("_")[1][:4]
            else:
                attr_short = attr[:4]

            file_path += f"{attr_short}={getattr(self, attr)}_"
        return file_path[:-1] + ".txt"


class PANConfig(PBM):
    split: str = Field(
        "train",
        description="Split to use for training",
        choices=["train", "val", "test"],
    )
    anonymizer: str = Field(
        "none",
        description="Anonymizer to use",
        choices=["none", "presidio"],
    )
    target: str = Field(description="target to predict", default="gender")
    total_queries: int = Field(
        ..., description="Total number of queries to be presented to the model"
    )
    data_dir: str = Field(
        "data/pan/2018",
        description="Path to the data directory",
    )
    subsample: int = Field(
        description="Number of examples to subsample from the dataset",
        default=0,
    )
    cot: bool = Field(False, description="Whether to use COT as part of the query")

    class Config:
        extra = Extra.forbid

    def get_filename(self) -> str:
        file_path = ""
        for attr in vars(self):
            if attr == "data_dir":
                continue
            file_path += f"{attr}={getattr(self, attr)}_"
        return file_path[:-1] + ".txt"


class HANDConfig(PBM):
    path: str = Field(
        "data/handcrafted/handcrafted.json",
        description="Path to the examples file",
    )

    class Config:
        extra = Extra.forbid

    def get_filename(self) -> str:
        file_path = ""
        for attr in vars(self):
            if attr == "path":
                continue
            file_path += f"{attr}={getattr(self, attr)}_"
        return file_path[:-1] + ".txt"


class REDDITConfig(PBM):
    path: str = Field(
        ...,
        description="Path to the file",
    )
    paths: List[str] = Field(
        default_factory=list,
        description="Paths to the files for merging",
    )
    outpath: str = Field(
        ...,
        description="Path to write to for comment scoring",
    )
    eval: bool = Field(
        default="False",
        description="Whether to only evaluate the corresponding profiles.",
    )
    eval_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Settings for evaluation.",
    )
    decider: str = Field(
        default="model", description="Decider to use in case there's no match."
    )
    profile_filter: Dict[str, int] = Field(
        default_factory=dict, description="Filter profiles based on comment statistics."
    )
    max_prompts: Optional[int] = Field(
        default=None, description="Maximum number of prompts asked (int total)"
    )
    header: Optional[str] = Field(default=None, description="Prompt header to use")
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt to use"
    )
    individual_prompts: bool = Field(
        False,
        description="Whether we want one prompt per attribute inferred or one for all.",
    )

    def get_filename(self) -> str:
        file_path = ""
        for attr in vars(self):
            if attr in ["path", "outpath"]:
                continue
            if attr == "profile_filter":
                file_path += (
                    str([f"{k}:{v}" for k, v in getattr(self, attr).items()]) + "_"
                )
            else:
                file_path += f"{attr}={getattr(self, attr)}_"
        return file_path[:-1] + ".txt"

    class Config:
        extra = Extra.forbid


class CHATConfig(PBM):
    n_rounds: int = Field(
        default=3,
        description="Number of generations. In each odd round the conversation starting bot is called \
            while in the even rounds the responding bot is called.",
    )

    mode: str = Field(
        default=None, description="Type mode: reasoning for a reasoning bot"
    )

    chat_starter: str = Field(
        default="investigator",
        description="The name of the bot starting the conversation.",
    )

    investigator_bot_system_prompt_path: str = Field(
        default="./data/curious_bots/system_prompts/1b_investigator_system_prompt.txt",
        description="Path to the file containing the investigator bot system prompt",
    )

    user_bot_system_prompt_path: str = Field(
        default="./data/curious_bots/system_prompts/1b_user_bot_system_prompt.txt",
        description="Path to the file containin the investigator bot system prompt",
    )

    guess_feature: str = Field(
        default="location",
        description="Indicate the feature that is to be guessed by the Investigator bot. Note that \
            the behavior will depend on the actual prompts passed to the bots in the respective fields, \
            merely, this field is used for organizing the saved logs in folders correctly.",
    )

    user_bot_personalities_path: str = Field(
        default="./data/curious_bots/user_bot_profiles.json",
        description="Path to the json file that stores the dictionary of the personalities",
    )

    user_bot_personality: int = Field(
        default=None,
        description="If this argument is set to an integer included in the .json containing the personalities, \
            then only this personality will be executed, otherwise, the whole range of personalities is iterated through.",
    )

    investigator_bot: ModelConfig = Field(
        default=None, description="Investigator model used in generation"
    )

    user_bot: ModelConfig = Field(
        default=None, description="Investigator model used in generation"
    )

    class Config:
        extra = Extra.forbid

    def get_filename(self) -> str:
        file_path = ""
        for attr in vars(self):
            if "path" in str(attr):
                filename_attr = (
                    str(getattr(self, attr)).replace("/", "_").replace(".", "")
                )
                file_path += f"{attr}={filename_attr}"
            else:
                file_path += f"{attr}={getattr(self, attr)}_"
        return file_path[:-1] + ".txt"

    class Config:
        extra = Extra.forbid


class CHAT_EVALConfig(PBM):
    chat_path_prefix: str = Field(
        default=None, description="Prefix of the path of the chats"
    )

    start: int = Field(
        default=None,
        description="The starting numeration of the conversation with the same hash (retries). If left unassigned, only the hashed name will be loaded.",
    )

    end: int = Field(
        default=None,
        description="The ending numeration of the conversatio0n with the same has (retries). If left unassigned, only the hashed name will be loaded.",
    )

    evaluation_model: ModelConfig = Field(
        default=None,
        description="The model config of the evaluator model, this model will match the answersextracted by the interpereter model to the ground truth",
    )

    use_model_aid: bool = Field(
        default=True,
        help="Set to False if you do not want to use the evaluation model to aid evaluation",
    )

    def get_filename(self) -> str:
        chat_prefix = (self.chat_path_prefix).split("/")[-1]
        file_path = f"{chat_prefix}_{self.start}_{self.end}_{self.use_model_aid}.txt"
        return file_path

    class Config:
        extra = Extra.forbid


class SYNTHETICConfig(PBM):
    hardnesses: list = Field(
        default=[5], description="Set the hardness of the generated comment"
    )

    personalities_path: str = Field(
        default=None, description="Path to the personalities of the user bots"
    )

    features: list = Field(
        default=["city_country"],
        description="The features on which to generate synthetic content",
    )

    max_trial: int = Field(
        default=None, description="Maximum trial after which there is no regeneration"
    )

    base_path: str = Field(default="./data/synthetic", description="")

    # -- Generation -- #
    investigator_bot: ModelConfig = Field(
        default=None, description="Investigator model used in generation"
    )

    user_bot: ModelConfig = Field(
        default=None, description="Investigator model used in generation"
    )

    # -- Evaluation -- #
    evaluation_model: ModelConfig = Field(
        default=None,
        description="The model config of the evaluator model, this model will match the answersextracted by the interpereter model to the ground truth",
    )

    def get_filename(self) -> str:
        file_path = ""
        for attr in vars(self):
            if "path" in str(attr):
                filename_attr = (
                    str(getattr(self, attr)).replace("/", "_").replace(".", "")
                )
                file_path += f"{attr}={filename_attr}"
            else:
                file_path += f"{attr}={getattr(self, attr)}_"
        return file_path[:-1] + ".txt"

    class Config:
        extra = Extra.forbid


class Config(PBM):
    # This is the outermost config containing subconfigs for each benchmark as well as
    # IO and logging configs. The default values are set to None so that they can be
    # overridden by the user
    output_dir: str = Field(
        default=None, description="Directory to store the results in"
    )
    seed: int = Field(default=42, description="Seed to use for reproducibility")
    task: Task = Field(
        default=None, description="Task to run", choices=list(Task.__members__.values())
    )
    task_config: (
        ACSConfig
        | PANConfig
        | HANDConfig
        | REDDITConfig
        | CHATConfig
        | CHAT_EVALConfig
        | SYNTHETICConfig
    ) = Field(default=None, description="Config for the task")
    gen_model: ModelConfig = Field(
        default=None, description="Model to use for generation, ignored for CHAT task"
    )
    store: bool = Field(
        default=True, description="Whether to store the results in a file"
    )
    save_prompts: bool = Field(
        False, description="Whether to ouput the prompts in JSON format"
    )
    dryrun: bool = Field(
        False, description="Whether to just output the queries and not predict"
    )
    timeout: int = Field(
        0.5, description="Timeout in seconds between requests for API restrictions"
    )

    def get_out_path(self, file_name) -> str:
        path_prefix = "results" if self.output_dir is None else self.output_dir

        if self.task.value == "CHAT":
            investigator_bot_name = self.task_config.investigator_bot.get_name()
            user_bot_name = self.task_config.user_bot.get_name()
            file_path = f"{path_prefix}/{self.task.value}/{investigator_bot_name}-{user_bot_name}/{self.seed}/{self.task_config.guess_feature}/{file_name}"
        elif self.task.value == "CHAT_EVAL":
            file_path = (
                "/".join((self.task_config.chat_path_prefix).split("/")[:-1])
                + "/"
                + file_name
            )
        else:
            model_name = self.gen_model.get_name()
            file_path = (
                f"{path_prefix}/{self.task.value}/{model_name}/{self.seed}/{file_name}"
            )

        return file_path
