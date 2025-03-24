<p align="center">
  <img src="assets/logo.png" alt="Alt text"/>
</p>

## Overview

Cactus Compute is on a mission to provide affordable, eco-friendly AI compute by harnessing the vast, untapped power of idle mobile devices. We connect everyday smartphone owners to teams, researchers, and businesses running ML workloads, resulting in lower costs, reduced carbon footprints, and a democratized AI ecosystem.

## Getting Started

1. Visit [cactuscompute.com](https://cactuscompute.com), sign up, and get your token.
2. Install the package:
   ```bash
   pip install cactus-lib
   ```
3. Set your token:
   ```bash
   export CACTUS_TOKEN=<your-token-from-the-website>
   ```
4. Initialize the model:
   ```python
   from cactus.nn import CactusLanguageModel
   from cactus.config import CactusConfig

   config = CactusConfig(
       model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
       lora_alpha=32,
       lora_r=8,
       lora_dropout=0.1,
   )

   model = CactusLanguageModel(config)

   # Train like a normal pytorch model
   ```
4. You can use Cactus functionalities:
   ```python
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
   ```

- **Demo:** [Watch our performance comparison against an NVIDIA A100 »](https://cactuscompute.com)

## Why Cactus Compute?
1. **Cost Savings:** Train and fine-tune ML models at a fraction of the cost of traditional cloud GPUs.  
2. **Sustainability:** Mobile GPUs are more energy-efficient than data-center GPUs, drastically reducing carbon footprints.  
3. **Accessibility:** We aim to empower smaller teams, individual researchers, and organizations with limited budgets to access robust AI compute.  
4. **Simplicity:** The Cactus Library seamlessly integrates with popular ML frameworks (PyTorch, TensorFlow, JAX) without complex cluster configurations.  
5. **Democratized Earnings:** Individuals earn passive income by securely sharing the idle capacity of their phones.


## How It Works
1. **Install the Cactus App (Providers):** Individuals install the app on their phone, which automatically contributes GPU/CPU cycles when the device is idle and charging.  
2. **Build with the Cactus Library (Consumers):** ML teams include the Cactus Library (`pip install cactus`) in their training scripts and simply point their training job to the Cactus network.  
3. **Automated Distribution:** Our backend, **Cactus Tango**, splits and routes jobs to available mobile devices, ensuring load balancing, fault tolerance, and efficient data transfer.  
4. **Optimized Execution:** **Cactus Ferra**, our proprietary kernel, accelerates deep learning workloads on mobile hardware, matching a single data-center GPU’s performance with a collection of consumer devices.

## Privacy & Security
- **Data Encryption:** End-to-end encryption in transit and at rest using TLS/SSL.  
- **Ephemeral Storage:** No training data or model weights are written to disk; everything is processed in volatile RAM.  
- **Randomized Task Allocation:** Individual devices receive small, randomized pieces of workloads, preventing reconstruction of any meaningful data.  

## Get Involved
### For Compute Providers
- **Sign up for Early Access:** [Join the waitlist »](https://cactuscompute.com)  
- **Install the Cactus App:** Coming soon on [Google Play](https://cactuscompute.com)  
- **Passive Earnings:** Earn money for every successful training epoch contributed.

### For Compute Consumers (ML Teams, Researchers, Businesses)
- **Early Beta Access:** [Request an invite »](https://cactuscompute.com)  
- **Documentation & Examples:** [Read the Docs »](#)  
- **Pricing:** Pay only for *runtime* (actual computation), typically at 50% below cloud GPU costs.

## Contributing
We welcome open-source contributions to the **Cactus Library** (the Python-based interface for distributing ML workloads). Whether you’re improving documentation, optimizing kernels, or adding new features, your help is invaluable.

1. Fork this repository.  
2. Create a feature branch (`git checkout -b feature/awesome-improvement`).  
3. Commit your changes (`git commit -m 'Add awesome feature'`).  
4. Push to the branch (`git push origin feature/awesome-improvement`).  
5. Open a Pull Request.  

## Community & Support
- **Slack/Discord:** [Coming soon](#) 
- **Email:** founders@cactuscompute.com  
- **Follow us on Twitter:** [@CactusCompute](#)

## License
- **Cactus Library:** Open source (Apache 2.0).
