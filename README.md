# LoRA-XS: **Lo**w-Rank Adaptation with Extremely Small Number of Parameters

We will update the README soon with instructions to setup the environment and to run experiments.

## Introduction
The recent trend in scaling language models has led to a growing demand for parameter-efficient tuning (PEFT) methods such as LoRA (Low-Rank Adaptation). LoRA consistently matches or surpasses the full fine-tuning baseline with fewer parameters. However, handling numerous task-specific or user-specific LoRA modules on top of a base model still presents significant storage challenges. To address this, we introduce LoRA-XS (**Lo**w-**R**ank **A**daptation with e**X**tremely **S**mall number of parameters), a novel approach leveraging Singular Value Decomposition (SVD) for parameter-efficient fine-tuning. LoRA-XS introduces a small r x r weight matrix between frozen LoRA matrices, which are constructed by SVD of the original weight matrix. Training only r x r weight matrices ensures independence from model dimensions, enabling more parameter-efficient fine-tuning, especially for larger models. LoRA-XS achieves a remarkable reduction of trainable parameters by over 100x in 7B models compared to LoRA. Our benchmarking across various scales, including GLUE, GSM8k, and MATH benchmarks, shows that our approach outperforms LoRA and recent state-of-the-art approaches like VeRA in terms of parameter efficiency while maintaining competitive performance.

[//]: # (## Quick Start)

[//]: # ()
[//]: # (Environment setup)

[//]: # (```)

[//]: # (pip install -r requirements.txt)

[//]: # (```)

[//]: # ()
[//]: # (Run experiments:)

[//]: # (```python)

[//]: # (python scripts/run_glue.py --target_task cola)

[//]: # (```)
