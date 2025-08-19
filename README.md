# When Punctuation Matters: A Large-Scale Comparison of Prompt Robustness Methods for LLMs

[![arXiv](https://img.shields.io/badge/cs.CL-arXiv%3A2508.11383-B31B1B.svg)](https://arxiv.org/abs/2508.11383)

Large Language Models (LLMs) are highly sensitive to subtle, non-semantic variations in prompt phrasing and formatting. In this work, we present the first systematic evaluation of 4 methods for improving prompt robustness within a unified experimental framework. We benchmark these techniques on 8 models from Llama, Qwen and Gemma families across 52 tasks from Natural Instructions dataset. Our evaluation covers robustness methods from both fine-tuned and in-context learning paradigms, and tests their generalization against multiple types of distribution shifts. Finally, we extend our analysis to GPT-4.1 and DeepSeek V3 to assess frontier models' current robustness to format perturbations. Our findings offer actionable insights into the relative effectiveness of these robustness methods, enabling practitioners to make informed decisions when aiming for stable and reliable LLM performance in real-world applications.

## Getting data

Clone `natural-instructions` so that they are located next to `prompt-instability`:
```
git clone git@github.com:allenai/natural-instructions.git
```

## Setup

```
cd prompt-instability

conda create -n aa --file environment.yml

conda activate aa

pip install trl==0.12.1 unsloth==2024.11.7 xformers==0.0.28.post3 bitsandbytes==0.44.1 wandb==0.18.7 dotenv
```

Prepare a directory with format train/test splits (this should run several minutes):
```
bash scripts/generate_test_splits.sh
```

Launch a single evaluation (several hours):
```
bash scripts/launch.sh 0 unsloth/Llama-3.2-1B-Instruct 2 random ---iid-no-chat-template 0
```
(refer to `scripts/launch.sh` to understand the order and meaning of arguments)

Launch sequential evaluation of original model, finetuning and evaluation of finetuned model (should be under 24 hours):
```
mkdir training

bash scripts/autoresearcher.sh 0 unsloth/Llama-3.2-1B-Instruct random ---iidx2-no-chat-template llama1b_iidx2
```
(refer to `scripts/autoresearcher.sh` to understand the order and meaning of arguments)
