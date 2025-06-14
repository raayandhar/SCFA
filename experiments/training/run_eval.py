import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

import os
import subprocess
from tqdm import tqdm

checkpoints_path = os.path.join(os.path.dirname(__file__), "checkpoints")
for checkpoint in tqdm(os.listdir(checkpoints_path)):
    tokenizer.save_pretrained(os.path.join(checkpoints_path, checkpoint))
    result = subprocess.run([
        "lm_eval", "--model", "hf",
        "--model_args", f"pretrained={checkpoints_path}/{checkpoint},dtype=bfloat16,trust_remote_code=True",
        "--tasks", "lambada_openai,hellaswag,piqa,boolq,winogrande,arc_easy,mmlu_abstract_algebra",
        "--device", "cuda:1",
        "--batch_size", "auto:8",
        "--wandb_args", "project=lm-eval-harness-integration",
        "--output_path", "eval_results",
        "--log_samples"
    ], capture_output=False, text=True)