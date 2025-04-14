<p align="center">
  <img src="assets/logo.png" alt="Alt text"/>
</p>

![License](https://img.shields.io/github/license/hmunachi/SuperLazyLanguageModel?style=flat-square)[![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com//company/80434055) [![Twitter](https://img.shields.io/twitter/follow/hmunachii?style=social)](https://twitter.com/hmunachii)

Author: [Henry Ndubuaku](https://www.linkedin.com/in/henry-ndubuaku-7b6350b8/)

## Overview

I mean, do not train or fine-tune LLMs on your laptop, traing is done at much higher precision than inference (float32 or bfload16). Also, additional memory is often used for the gradients, optimizer states, and batch size. So, 4 - 6x the model size. For simplicity, around 8-24G of RAM per 1B params. 

HOWEVER, if you must do so on a laptop for whatever weird reason, this library implements most language models such that only the weights for each layer is loaded to the RAM, it implements LoRA fine-tuning such that frozen params are memory-mapped rather than loaded.

Note the following:
1) Compute intensity = computation time / communication time, and maximisin this means maximising GPU utilisation. 
2) Many computations in transformer models can be parallelised, QKV projections for example. 
3) Most operations in transformers follow the signature A @ B * Scale, A.K.A scaled dot-product. 
4) Q @ K.T / sqrt(dimK) is obiously equivalent to Q @ K.T * dimK^(-1/2)
5) But Lora_A @ Lora_B = Lora_A @ Lora_B * 1, also A * B = I @ A * B, and so on.

We expressed the transformer forward pass and the backward vector-jacobian products for each layer as a bunch of scaled matmuls, which are bundled together and executed in parallel across different CPU cores as C++ extensions to bypass GIL. This concept makes it easy for an upcoming feature, where each bundle could be distributed across your friends' laptops, such that they only execute one operation called Bundled Scaled Matmul. You're welcome.

## Limitations 

1) Gradient accumulation, gradient checkpointing and lazy execution trade time-complexity for memory-efficiency, but you have no choice, do you?
2) Yeah...your laptop will definitley heat up, GPUs burn up at data centers and cost so much to cool, your laptop is not special. 

## Supported Models 

- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- Qwen/Qwen2.5-0.5B
- Qwen/Qwen2.5-0.5B-Instruct
- Qwen/Qwen2.5-1.5B
- Qwen/Qwen2.5-1.5B-Instruct
- Qwen/Qwen2.5-3B
- Qwen/Qwen2.5-3B-Instruct

## Getting Started

1. ```bash
   pip install sllm-lib
   ```
2. Initialize the model:
   ```python
   from sllm.nn import SuperLazyLanguageModel
   from sllm.config import Config

   config = Config(
       model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
       lora_alpha=32,
       lora_r=8,
       lora_dropout=0.1,
   )

   model = SuperLazyLanguageModel(config)

   # Train like a normal pytorch model
   ```
4. You can use SLLM functionalities:
   ```python
   import torch
   from datasets import load_dataset

   from sllm.nn import SuperLazyLanguageModel
   from sllm.train import sft, prepare_dataset

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

   model = SuperLazyLanguageModel(
      name=name, 
      lora_alpha=32, 
      lora_r=8, 
      lora_dropout=0.1,
   )

   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
   sft(model=model, dataset=dataset, optimizer=optimizer, batch_size=8, epochs=3)
   ```

## Contributing
Whether you’re improving documentation, optimizing kernels, or adding new features, your help is invaluable.

1. Create a feature branch (`git checkout -b feature/awesome-improvement`).  
2. Commit your changes (`git commit -m 'Add awesome feature'`).  
3. Push to the branch (`git push origin feature/awesome-improvement`).  
4. Open a Pull Request.  
