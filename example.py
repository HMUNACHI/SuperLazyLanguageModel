import torch
from datasets import load_dataset

from cactus.nn import CactusLanguageModel
from cactus.train import sft, prepare_dataset

torch.manual_seed(42)

name = "Qwen/Qwen2-0.5B-Instruct"
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:200]")

dataset = prepare_dataset(
    model_name=name, 
    instructions=dataset["instruction"], 
    responses=dataset["output"], 
    inputs=dataset["input"],
    max_seq_len=256,
)

model = CactusLanguageModel(
    name=name, 
    lora_alpha=32, 
    lora_r=8, 
    lora_dropout=0.1,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
sft(model=model, dataset=dataset, optimizer=optimizer, batch_size=8, epochs=3)