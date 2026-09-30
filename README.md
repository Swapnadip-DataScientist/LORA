# TinyLlama Fine-Tuning with LoRA and PEFT

This project demonstrates how to fine-tune a language model efficiently using **LoRA (Low-Rank Adaptation)** and **PEFT (Parameter-Efficient Fine-Tuning)**.

Instead of retraining all model parameters, LoRA injects small trainable low-rank matrices into selected transformer layers while keeping the original base model weights frozen.

The objective of this project is to fine-tune **TinyLlama-1.1B-Chat-v1.0** on the **Databricks Dolly-15K** instruction-following dataset and compare the model's behavior before and after fine-tuning.

---

## Project Objective

Traditional full fine-tuning can be expensive because it requires updating a very large number of model parameters.

LoRA reduces this cost by training only a small set of additional parameters.

This project covers:

- Base model evaluation
- Dolly dataset preparation
- Chat-format conversion
- LoRA configuration
- LoRA adapter injection
- Supervised fine-tuning using `SFTTrainer`
- Mixed-precision training
- Gradient checkpointing
- Adapter persistence
- Base vs fine-tuned model evaluation

---

## Architecture

```text
Databricks Dolly-15K
        |
        v
Dataset Formatting
        |
        v
TinyLlama-1.1B-Chat
        |
        v
LoRA Adapter Injection
(q_proj + v_proj)
        |
        v
Supervised Fine-Tuning
        |
        v
Trained LoRA Adapter
        |
        v
Base vs Fine-Tuned Evaluation


## Technology Stack
- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- TRL
- Hugging Face Datasets
- Accelerate
- Google Colab GPU

#Base Model

TinyLlama/TinyLlama-1.1B-Chat-v1.0
TinyLlama is used as the base causal language model for the experiment.

Dataset
The project uses:
databricks/databricks-dolly-15k