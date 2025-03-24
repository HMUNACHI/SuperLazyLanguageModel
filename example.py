import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from cactus.nn import CactusLanguageModel
from cactus.config import CactusConfig

torch.manual_seed(42)

def format_example(example):
    prompt_text = f"Instruction: {example['instruction']}\n"
    if example['input']:
        prompt_text += f"Input: {example['input']}\n"
    prompt_text += "Response:"
    full_text = prompt_text + " " + example["output"]
    return {"text": full_text, "prompt_text": prompt_text}

def tokenize_fn(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_len,
    )
    prompt_tokens = tokenizer(
        example["prompt_text"],
        truncation=False,
        add_special_tokens=False,
    )["input_ids"]
    tokenized["prompt_length"] = len(prompt_tokens)
    return tokenized

def mask_labels(example):
    labels = example["input_ids"].copy()
    prompt_length = example["prompt_length"]
    for i in range(min(prompt_length, len(labels))):
        labels[i] = -100
    example["labels"] = labels
    return example

batch_size = 2
max_seq_len = 512
model_name = "Qwen/Qwen2-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
formatted_dataset = dataset.map(format_example)
tokenized_dataset = formatted_dataset.map(tokenize_fn)
tokenized_dataset = tokenized_dataset.map(mask_labels)

input_ids = torch.tensor(tokenized_dataset["input_ids"])
attention_masks = torch.tensor(tokenized_dataset["attention_mask"])
labels = torch.tensor(tokenized_dataset["labels"])

num_tokens = input_ids.numel() // 1000
print(f"Training on {num_tokens}k tokens")

input_batches = input_ids.split(batch_size)
mask_batches = attention_masks.split(batch_size)
label_batches = labels.split(batch_size)

config = CactusConfig(
    model_name=model_name,
    lora_alpha=32,
    lora_r=8,
    lora_dropout=0.1,
)

model = CactusLanguageModel(config)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
epochs = 10

for epoch in range(epochs):
    epoch_loss = 0
    start = time.time()
    model.train()

    for batch_input, batch_mask, batch_labels in zip(input_batches, mask_batches, label_batches):
        
        optimizer.zero_grad()
        output = model(input_ids=batch_input, attention_mask=batch_mask, labels=batch_labels)
        loss = output.loss
        loss.backward()
        print(loss.item())
        
        optimizer.step()
        epoch_loss += loss.item()
        
    print(f"Epoch {epoch+1}: loss: {epoch_loss / len(input_batches):.4f} | Time: {time.time() - start:.2f} seconds")
