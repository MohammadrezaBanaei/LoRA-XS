# Code based on https://github.com/GraphPKU/PiSSA script with minimal changes.

import copy
import os
import yaml
import json
import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Literal

import torch
import transformers
from transformers import Trainer, set_seed
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from utils.initialization_utils import find_and_initialize


IGNORE_INDEX = -100
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    adapter_name_or_path: Optional[str] = field(default=None)
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_split: str = field(
        default="train[:100000]", metadata={"help": "(`['train', 'test', 'eval']`):"}
    )
    dataset_field: List[str] = field(
        default=None, metadata={"help": "Fields of dataset input and output."}
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={
        "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}, )
    lora_r: int = field(default=None, metadata={
        "help": "The rank of the low-rank adapter."})


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    set_seed(script_args.seed)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        device_map="auto",
    )

    if script_args.lora_r is not None:
        lora_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_r,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    else:
        raise ValueError("LoRA rank should be provided.")

    now = datetime.datetime.now()
    now.strftime('%Y-%m-%dT%H:%M:%S') + ('-%02d' % (now.microsecond / 10000))

    adapter_name = "default"
    peft_config_dict = {adapter_name: lora_config}

    with open("config/reconstruct_config.yaml", 'r') as stream:
        reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)
    reconstr_type = reconstr_config['reconstruction_type']
    reconstr_config[reconstr_type]['rank'] = peft_config_dict[adapter_name].r

    script_args.output_dir = f"{script_args.output_dir}/{script_args.model_name_or_path}/" \
                             f"{script_args.data_path}_split_{script_args.dataset_split}/" \
                             f"LoRA_init_{reconstr_type}_rank_{peft_config_dict[adapter_name].r}_lr_" \
                             f"{script_args.learning_rate}_seed_{script_args.seed}/output_{now}"
    os.makedirs(script_args.output_dir)

    with open(os.path.join(script_args.output_dir, 'reconstr_config.json'), 'w') as fp:
        json.dump(reconstr_config, fp)

    find_and_initialize(model, peft_config_dict, adapter_name=adapter_name, reconstr_type=reconstr_type,
                        writer=None, reconstruct_config=reconstr_config)

    for param in model.parameters():
        param.data = param.data.contiguous()

    model.print_trainable_parameters()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    raw_train_datasets = load_dataset(script_args.data_path, split=script_args.dataset_split)
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": script_args.dataset_field[0],
                   "response": script_args.dataset_field[1]}
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, data_collator=data_collator)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)
    model.config.use_cache = False
    trainer.train()
    trainer.save_state()
    model.save_pretrained(os.path.join(script_args.output_dir, 'ft'))


if __name__ == "__main__":
    train()
