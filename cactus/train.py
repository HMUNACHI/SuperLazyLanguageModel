import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import time  # add time module if not imported

from cactus.utils import clear_gradient_dir
from cactus.common import MINI_BATCH_SIZE


def format_example(example):
    prompt_text = f"Instruction: {example['instruction']}\n"
    if example["input"]:
        prompt_text += f"Input: {example['input']}\n"
    prompt_text += "Response:"
    full_text = prompt_text + " " + example["output"]
    return {"text": full_text, "prompt_text": prompt_text}


def prepare_dataset(model_name, instructions, responses, inputs=None, max_seq_len=256):

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

    dataset = Dataset.from_dict(
        {
            "instruction": instructions,
            "input": inputs,
            "output": responses,
        }
    )
    dataset = dataset.map(format_example)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = dataset.map(tokenize_fn)
    dataset = dataset.map(mask_labels)

    dataset = dataset.shuffle(seed=42)

    input_ids = torch.tensor(dataset["input_ids"])
    attention_masks = torch.tensor(dataset["attention_mask"])
    labels = torch.tensor(dataset["labels"])
    print(f"Training on {input_ids.numel() // 1000}k tokens")
    return {
        "input_ids": input_ids.split(MINI_BATCH_SIZE),
        "attention_mask": attention_masks.split(MINI_BATCH_SIZE),
        "labels": labels.split(MINI_BATCH_SIZE),
    }


def sft(model, dataset, optimizer, batch_size=1, epochs=1):
    grad_accum_steps = batch_size // MINI_BATCH_SIZE
    if grad_accum_steps < 1:
        grad_accum_steps = 1

    total_batches = len(dataset["input_ids"])
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        optimizer.zero_grad()
        clear_gradient_dir()
        accum_steps = 0

        epoch_start_time = time.time()

        with tqdm(total=total_batches, desc=f"Epoch {epoch+1}") as pbar:
            zipped_dataset = zip(
                dataset["input_ids"], dataset["attention_mask"], dataset["labels"]
            )
            for i, batch in enumerate(zipped_dataset, start=1):
                batch_input, batch_mask, batch_labels = batch
                output = model(
                    input_ids=batch_input,
                    attention_mask=batch_mask,
                    labels=batch_labels,
                )
                loss = output.loss
                loss = loss / grad_accum_steps
                loss.backward()
                epoch_loss += loss.item() * grad_accum_steps
                accum_steps += 1

                if accum_steps % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_steps = 0

                avg_loss = epoch_loss / i
                elapsed = time.time() - epoch_start_time
                sec_per_sample = elapsed / (i * MINI_BATCH_SIZE)
                sec_per_epoch = time.time() - epoch_start_time
                pbar.set_postfix(loss=f"{avg_loss:.1f}", sec_per_sample=f"{sec_per_sample:.2f}", sec_per_epoch=f"{sec_per_epoch:.2f}")
                pbar.update(1)

            if accum_steps > 0:
                optimizer.step()
                optimizer.zero_grad()