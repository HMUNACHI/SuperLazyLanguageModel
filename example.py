import torch
from datasets import load_dataset

from cactus.nn import CactusLanguageModel
from cactus.train import prepare_dataset, sft

torch.manual_seed(42)

name = "Qwen/Qwen2-0.5B-Instruct"
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:20]")

dataset = prepare_dataset(
    model_name=name,
    instructions=dataset["instruction"],
    responses=dataset["output"],
    inputs=dataset["input"],
    max_seq_len=256,
)

model = CactusLanguageModel(
    name=name,
    lora_alpha=16,
    lora_r=4,
    lora_dropout=0.1,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
sft(model=model, dataset=dataset, optimizer=optimizer, batch_size=4, epochs=3)
