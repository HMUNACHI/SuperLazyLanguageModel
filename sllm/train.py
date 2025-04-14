import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import time  # add time module if not imported

from sllm.utils import clear_gradient_dir
from sllm.common import MINI_BATCH_SIZE


def format_example(example):
    """
    Format a single example into a prompt and full training text.

    The function builds a prompt with a header (containing the instruction and, if available,
    additional input) followed by a "Response:" marker. It then concatenates the prompt with
    the output to form the full text for training.

    Args:
        example (dict): A dictionary with keys 'instruction', 'input', and 'output'.

    Returns:
        dict: A dictionary with keys:
            - "text": A string that concatenates the prompt and the output.
            - "prompt_text": The text before adding the output.
    """
    prompt_text = f"Instruction: {example['instruction']}\n"
    if example["input"]:
        prompt_text += f"Input: {example['input']}\n"
    prompt_text += "Response:"
    full_text = prompt_text + " " + example["output"]
    return {"text": full_text, "prompt_text": prompt_text}


def prepare_dataset(model_name, instructions, responses, inputs=None, max_seq_len=256):
    """
    Prepare and tokenize a dataset for training.

    The function creates a Hugging Face Dataset from the provided instructions, responses, and optional inputs.
    It then formats each example using `format_example`, tokenizes the text (padding/truncating to `max_seq_len`),
    and creates a special label mask in which the tokens corresponding to the prompt are replaced by -100 (to be ignored 
    during loss computation). Finally, the tokenized dataset is split into mini-batches.

    Args:
        model_name (str): The pretrained model name used to load the tokenizer.
        instructions (List[str]): A list of instruction strings.
        responses (List[str]): A list of response strings.
        inputs (List[str] or None, optional): A list of additional input strings for examples; defaults to None.
        max_seq_len (int, optional): Maximum sequence length for tokenization; defaults to 256.

    Returns:
        dict: A dictionary with keys 'input_ids', 'attention_mask', and 'labels', 
              where each value is a list of tensors split into mini-batches of size MINI_BATCH_SIZE.
    """
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
        # Copy the tokenized input_ids to labels.
        labels = example["input_ids"].copy()
        prompt_length = example["prompt_length"]
        # Mask the tokens corresponding to the prompt (set to -100) so that they do not contribute to the loss.
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
    """
    Perform supervised fine-tuning (SFT) on a language model.

    This function trains the model on the provided dataset using gradient accumulation.
    The effective batch size is determined as the product of MINI_BATCH_SIZE and grad_accum_steps.
    
    Gradient Accumulation Derivation:
        If the provided batch_size is larger than MINI_BATCH_SIZE (the size of each mini-batch),
        the gradients of several forward/backward passes are accumulated before performing an optimizer step.
        In order to ensure that the effective gradient is equivalent to that computed on the entire batch,
        the loss of each mini-batch is divided by grad_accum_steps. That is,
            loss_effective = (loss_mini_batch / grad_accum_steps)
        This scaling ensures that when the gradients from grad_accum_steps mini-batches are summed,
        the resulting update is equivalent to the gradient of the average loss over the full batch.
    
    During training, the function reports the average loss, seconds per sample, and seconds elapsed per epoch.

    Args:
        model (torch.nn.Module): The model to be fine-tuned.
        dataset (dict): A dictionary with keys 'input_ids', 'attention_mask', and 'labels',
            where each value is a list of tensors representing mini-batches.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        batch_size (int, optional): The total batch size for each optimizer update; defaults to 1.
        epochs (int, optional): Number of training epochs; defaults to 1.

    Returns:
        None
    """
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
                # Scale the mini-batch loss to average over grad_accum_steps
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
                pbar.set_postfix(
                    loss=f"{avg_loss:.1f}",
                    sec_per_sample=f"{sec_per_sample:.2f}",
                    sec_per_epoch=f"{sec_per_epoch:.2f}",
                )
                pbar.update(1)

            # Apply any remaining accumulated gradients.
            if accum_steps > 0:
                optimizer.step()
                optimizer.zero_grad()
