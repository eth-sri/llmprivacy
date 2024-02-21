# Overview

This is the repository accompanying our ICLR 2024 paper ["Beyond Memorization: Violating Privacy via Inference with Large Language Models"](https://arxiv.org/abs/2310.07298).

## Getting started

The easiest way to get started with the project is to simply run the `demo.py` script. This will load a synthetic example and run a single query on it. This will give you a brief overview of the the main prompt and outputs.

Before running the demo, you need to install the environment. We recommend using mamba to install the environment via `mamba env create -f environment.yaml` which will create an environment called `beyond-mem` (see for detailed instructions below). You can then activate the environment via `conda activate beyond-mem`.

The demo script uses the OpenAI API and you need to set the credentials in `credentials.py` to use the OpenAI-API (we provide a template in `credentials_template.py`). You can adapt the code directly in the `demo.py` file to use a different model (Line 21) and refer you to exemplarary configs such as `configs/reddit/running/reddit_llama2_7b.yaml` for reference.

If you want to run other experiments, you can use the `main.py` script. This script takes a config file as input and runs the respective experiment. We provide a set of sample configs in the `configs` folder. You can run the script via `python ./main.py --config_path <your_config>`. For more detail we refer to the documentation below.

## Structure

This repo contains both the original code for the experiments as well as the code used to create and evaluate synthetic examples. Please note that due to privacy concerns the original dataset will not be released.

We now describe the project structure in more detail

You can find all experimental configurations in `/configs`.

All datasets are stored in `/data` (Note that for the full set of synthetic examples refer the the description below).
- `/data/curios_bots` contains our data for the adversarial interaction setting.
- `/data/acs_dataset.py` contains the ACS dataset.
- `/data/pan` contains the structure for the PAN dataset. Note that the PAN dataset is not publicly available and we refer to the respective instructions in the folder.
- `/data/reddit` contains the structure for the Personal Reddit dataset. Note that the PR dataset is not publicly available and we refer to the respective synthetic example instructions below.
- `/data/synthetic` contains all synthetic examples released with the paper. We give a more detailed description below. The primariy file here is `data/synthetic/synthetic_dataset.jsonl`.

`/parsing` contains the code to parse the ACS Adult dataset.

`/scripts` contains scripts:
- `reparse_and_fix.sh` is used to fix the formatting issues of weaker language models


`/src` Contains the code for all our experiments. They are largely split into individual datasets with some shared code:
- `/acs` Contains all code for the ACS ablation in our Appendix.
- `/chat` Contains all code for the adversarial interaction setting.
- `/configs` Contains the programmatic configuration definitions. THIS IS WHERE YOU SHOULD LOOK IF ANY CONFIG FIELDS ARE UNCLEAR.
- `/models` Contains the code for the models used in the experiments.
  - `/models/model.py` Contains the abstract base class for all models.
  - `/models/model_factory.py` Contains the factory to create the models.
  - `/models/gcp.py` Contains the code for the Google models, in particular PaLM-2.
  - `/models/poe.py` Contains the code for Claude models, which at the time of project were not avaialable outside of the US and where accessed via a wrapper of Poe. Note that this is not the case anymore and we recommend using Google Vertex AI for this. Further note that the wrapper may break with any change of the Poe website.
  - `/models/hf_model.py` Contains the code for the HuggingFace models.
  - `/models/open_ai.py` Contains the code for all OpenAI models.
  - We also include several other models (`chain_model`, `multi_model`), which are not used in the paper, for completeness.
- `/pan` Contains all code for the PAN experiments.
- `/reddit` Contains all code for the PR experiment in our main paper.
- `/synthetic` Contains all code for the synthetic examples and evaluation in our main paper.
- `/utils` Contains common utility functions.
- `/visualization` Contains the code for the visualization of the results.


`/credentials.py` Contains the credentials for the OpenAI as well as Azure API.

`/environment.yaml` Contains the environment definition for the project.

`/main.py` Contains the main entry point for the experiments.

`/demo.py` Contains a simple demo to run a single query on the synthetic data.

`/all_plots.sh` Contains the script to generate all plots.


## Setup

Install mamba via:

```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

This can be used to install our environment via `mamba env create -f environment.yaml` which will create an environment called `beyond-mem`.


Assuming you have all data for your respective runs you can set the credentials in `credentials.py` to use the OpenAI-API. For HuggingFace models please also login first using `huggingface-cli login` and your respective API token. FOR Google Vertex-AI via GCP we refer to the respective setup instructions here [https://cloud.google.com/vertex-ai/docs?hl=en] 


## Running

To run any experiment proceed as follows:

```bash
conda activate beyond-mem
python ./main.py --config_path <your_config>
```

we provide a set of configs in the `configs` folder (Note that for many you have to adjust the data - `configs/reddit/running/reddit_synthetic_gpt4.yaml` should run all synthetic samples using OpenAIs GPT-4). We also provide a simple `demo.py` file which runs a single query from the synthetic data. Note that by default configs will redirect the output to the `./results` folder. You can change this by adapting the `output_dir` in the config.

### Running on synthetic samples

Configs generally require a datasource - as PR is not publicly available due to Ethics- and Privacy-Guidelines, you can run a config on the respective synthetic datapoints as it is shown in `configs/reddit/running/reddit_synthetic_gpt4.yaml`. For this simply adapt the corresponding path in the config to the location of the synthetic dataset (contained in this repo).

### ACS

The respecitve ACS ablation experiments we ran the configs contained in `configs/acs` in particular `acs_{gpt4|xgb}_{1..5}`.


### PAN 

Our respective PAN experiment can be found in `configs/pan`. Note that the PAN dataset is access only (we left corresponding instructions in the respective folder). Depending on the used model (and its context size) you might want to subsample the number of comments per profile (via the subsample option the config).


### Running synthetic conversations

To generate adversarial conversations between bots, run `python main.py --config_path configs/4R_config_all_pers_location_age_sex.yaml`. For evaluation, run `python main.py --config_path configs/4R_config_all_pers_location_age_sex_eval.yaml`.


### Creation of synthetic examples

First, to generate a `.json` with the synthetic profiles run `python user_bot_profiles.py` on the path `data/curious_bots`. 
Note that the synthetic example generation is seeded with real-world examples that we do not intend to release for ethical considerations. Therefore, before synthetic samples can be created, one has to construct examples for both the investigator and the user, and for each hardness level and feature. The examples have to be placed on the paths: `data/synthetic/examples/<role>/<feature>/<role>_examples_<feature>_hard<hardness_level>.txt`, where `<role>` may be `investigator` or `user`, `feature` is the feature name as in the profiles, and `hardness_level` is an integer between 1 and 5.

As the examples for the `education` feature a completely manually constructed, we publish them, and they can serve as a template for how the examples for other features are to be consturcted.

Once all examples have been constructed, run the command `python main.py --config_path configs/synthetic_data/synthetic.yaml`. Note that even given our original examples, this would generate a superset of our released synthetic examples, containing also different hardness scores. This is because the released examples have been filtered and adjusted for appropriate hardness by hand.

If you have constructed examples only for a subset of the features or hardness levels, you may reduce the hardness level or feature coverage in `configs/synthetic_data/synthetic.yaml`.


## Plotting

All plots are generated using the `src/visualization/visualize_reddit.py` script. In particular all main plots were generated using the `all_plots.sh` script. In case you want to make different plots, we recommend that you take a look a these files and adapt them to your corresponding needs. The corresponding primary files are `src/visualization/visualize_acs.py` for the ACS plots and `src/visualization/visualize_reddit.py` for all plots on PR.

# [25.01.24] Synthetic Examples
We release the synthetic examples that are part of our evaluation in [the paper](https://arxiv.org/abs/2310.07298v1). For a more detailed explanation of the example creation, we refer to the paper appendix. Additionally, we want to make the following relevant disclaimers and notes about the examples:

- While each conversation is seeded with a complete synthetic profile, the evaluation per profile is only on one of the profile features (denoted by "feature").
- Each conversation consists of one question asked and the (synthetic) user answering the question.
- These examples are qualitative replacements for some examples obversed in the PR dataset and not statistically representative for (1) real online texts and (2) do not reflect the distribution in PR itself. While we have (and continue) put a lot of effort in creating samples we do not provide any guarantees on them.