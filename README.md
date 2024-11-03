# LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters

Code for the paper: "[LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters](https://arxiv.org/abs/2405.17604)"

## Introduction
We introduce LoRA-XS (**Lo**w-**R**ank **A**daptation with e**X**tremely **S**mall number of parameters), a novel approach leveraging Singular Value Decomposition (SVD) for parameter-efficient fine-tuning. LoRA-XS introduces a small r x r weight matrix between frozen LoRA matrices, which are constructed by SVD of the original weight matrix. Training only r x r weight matrices ensures independence from model dimensions, enabling more parameter-efficient fine-tuning, especially for larger models. LoRA-XS achieves a remarkable reduction of trainable parameters by over 100x in 7B models compared to LoRA. Our benchmarking across various scales, including GLUE, GSM8k, and MATH benchmarks, shows that our approach outperforms LoRA and recent state-of-the-art approaches like VeRA in terms of parameter efficiency while maintaining competitive performance.


<p align="center">
  <img src="./assets/LoRA_versus_LoRAxs.png" alt=“LoRA-XS” width=90%>
  <br> Visual comparison of LoRA and <b>LoRA-XS</b> techniques. The key distinction of LoRA-XS lies in its use of a small<br> trainable matrix <b>R</b> between frozen low-rank matrices A and B derived from truncated SVD of pretrained weights.
</p>
  

## Requirements
We recommend running the scripts inside a conda environment.
You can run the following commands to create the conda environment, as well as installing needed libraries:
```bash
git clone https://github.com/MohammadrezaBanaei/LoRA-XS.git
conda create -n loraxs python=3.8.13
conda activate loraxs
cd LoRA-XS ; pip install -r requirements.txt
```
## Quickstart
LoRA-XS is built on top of HuggingFace Transformers and PEFT libraries. As demonstrated below, LoRA modules are first
added to the model as usual. Then, the `find_and_initialize` function will go through LoRA modules and transform each
to a **LoRA-XS** module, which involves computing a truncated SVD of the pretrained weights for the frozen A/B matrices,
as well as injecting r*r matrix (i.e., matrix **R** in the above figure) into the module.
The only needed file is a config file indicating the needed SVD arguments for initialization which is provided in the
`config` directory.
```bash
from peft import LoraConfig, get_peft_model
from utils.initialization_utils import find_and_initialize  # used to transform LoRA to LoRA-XS 
config = LoraConfig(
    r=lora_rank,
    target_modules=lora_target_modules,
    task_type="CAUSAL_LM", # assuming a decoder-only model in this example
        )
model = get_peft_model(model, config)

with open("config/reconstruct_config.yaml", 'r') as stream:
    reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)
    
adapter_name = "default"  # assuming a single LoRA adapter per module should be transformed to LoRA-XS
peft_config_dict = {adapter_name: lora_config}

# specifying LoRA rank for the SVD initialization
reconstr_config['svd']['rank'] = lora_rank
    
find_and_initialize(
    model, peft_config_dict, adapter_name=adapter_name, reconstr_type='svd',
    writer=None, reconstruct_config=reconstr_config
    )

# perform training...

# LoRA-XS can be merged into the base model using `merge_and_unload` functionality of PEFT
model = model.merge_and_unload() 


```
## GLUE Experiments
**Note**: Feel free to limit the grid search in the following scripts if you want to train the model with a specific hyperparameter.
### Training from scratch
To reproduce our GLUE results for CoLA, SST-2 and QNLI tasks, please run the `scripts/run_glue.py` script as follows (using QNLI dataset as an example):
```bash
python scripts/run_glue.py --target_task qnli
```
### Training from MNLI-tuned models
Similar to previous work, the GLUE experiments on MRPC, RTE and STS-B tasks are initialized from an MNLI-tuned model.
Please run the `scripts/run_glue_pretrained.py` script as follows (using MRPC dataset as an example).
Please note that you need to put your MNLI-tuned models in the `model_checkpoints` directory before running the script. We provide MNLI-tuned (using LoRA-XS) checkpoints with various ranks for the RoBERTa-large model [here](https://drive.google.com/drive/folders/1qGeAvSvG-iRhTopyhIhi55LIUoRSsMob?usp=share_link).
```bash
python scripts/run_glue_pretrained.py --target_task mrpc
```
### Random versus SVD-based initialization
In order to run the LoRA-XS training with a random initialization (instead of the SVD-based initialization),
please run the following script (using QNLI as an example):
```bash
python scripts/run_glue_no_svd.py --target_task qnli
```
## Instruction Tuning Experiments
In order to run the instruction tuning experiments in the paper, please have a look at the following sections.
### Instruction Tuning for Mathematical Reasoning
In these set of experiments, the model is first trained on the MetaMathQA dataset
and then evaluated on GSM8K and MATH benchmarks.
Please run the following bash script for fine-tuning and evaluation of a decoder-only model.
If you want to fine-tune a different pre-trained model (current default is the Mistral-7B model),
feel free to change the `BASE_MODEL` variable in the `scripts/run_math_tuning.sh` script.
```bash
bash scripts/run_math_tuning.sh 
```
### Instruction Tuning for Commonsense Reasoning
#### Commonsense Data
In order to run the commonsense experiments please download the necessary data as follows.

First, download the fine-tuning [dataset](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json)
and put it in the `data/commonsense` directory.
For evaluation datasets, you can download needed evaluation dataset from
[here](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset)
and then put each dataset into its respective directory in `data/commonsense` .
#### Running Experiments
In these set of experiments, the model is first trained on a mixture of commmonsense reasoning datasets, and then separately
evaluated on eight commonsense datasets.
In order to perform fine-tuning, please run the following bash script.
If you want to fine-tune a different pre-trained model (current default is the LLaMA-3 model),
feel free to change the `BASE_MODEL` variable in the `scripts/run_commonsense_tuning.sh` script. 
```bash
bash scripts/run_commonsense_tuning.sh LORA_RANK LORA_ALPHA OUTPUT_MODEL_PATH 0
```
A typical fine-tuning experiment can be done with `LORA_RANK` of 32
and `LORA_ALPHA` of 64.
Once the model is fine-tuned, pass the desired LoRA-XS checkpoint to the
`scripts/run_commonsense_evaluate.sh` bash script for model evaluation.
For instance, for evaluating a LLaMA-3 model with a desired LoRA-XS checkpoint,
please run the script as:
```bash
bash scripts/run_commonsense_evaluate.sh meta-llama/Meta-Llama-3-8B PATH_TO_LORAXS_CHECKPOINT 0
```
## Citation
If you use this code for your research, please cite the following paper:
```
@article{balazy2024lora,
  title={LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters},
  author={Ba{\l}azy, Klaudia and Banaei, Mohammadreza and Aberer, Karl and Tabor, Jacek},
  journal={arXiv preprint arXiv:2405.17604},
  year={2024}
}
```

