
# All baseline plots
python src/visualization/visualize_reddit.py --path eval_results/full_eval_model_human.jsonl --folder full --plot_stack --complete
# HArdness and less precise
python src/visualization/visualize_reddit.py --path eval_results/full_eval_model_human.jsonl --folder less_precise_plots --plot_hardness --show_less_precise --min_certainty 3
# Drop plot
python src/visualization/visualize_reddit.py --path eval_results/full_eval_model_human.jsonl eval_results/gpt_4_anonymized_hard_04.jsonl --plot_drop --model GPT-4 --folder gpt4_drop

# Hardness 4 models plot
python src/visualization/visualize_reddit.py --path eval_results/full_eval_model_human.jsonl --folder hardness_4 --plot_hardness --models Llama-2-70b "PaLM 2 Chat" Claude-2 GPT-4

# Attribute accuracy plots
python src/visualization/visualize_reddit.py --path eval_results/full_eval_model_human.jsonl --folder attributes --plot_attributes --model GPT-4
python src/visualization/visualize_reddit.py --path eval_results/full_eval_model_human.jsonl --folder attributes --plot_attributes --model GPT-3.5
python src/visualization/visualize_reddit.py --path eval_results/full_eval_model_human.jsonl --folder attributes --plot_attributes --model Llama-2-7b
python src/visualization/visualize_reddit.py --path eval_results/full_eval_model_human.jsonl --folder attributes --plot_attributes --model Llama-2-13b
python src/visualization/visualize_reddit.py --path eval_results/full_eval_model_human.jsonl --folder attributes --plot_attributes --model Llama-2-70b
python src/visualization/visualize_reddit.py --path eval_results/full_eval_model_human.jsonl --folder attributes --plot_attributes --model "PaLM 2 Chat"
python src/visualization/visualize_reddit.py --path eval_results/full_eval_model_human.jsonl --folder attributes --plot_attributes --model "PaLM 2 Text"
python src/visualization/visualize_reddit.py --path eval_results/full_eval_model_human.jsonl --folder attributes --plot_attributes --model Claude-2
python src/visualization/visualize_reddit.py --path eval_results/full_eval_model_human.jsonl --folder attributes --plot_attributes --model Claude-Instant