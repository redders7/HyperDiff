import os
import yaml
from argparse import ArgumentParser

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import load_dataset
import utils
from math import ceil


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def load_model_and_tokenizer(config, config_dict):
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name_or_path,
        use_auth_token=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.generation_config.do_sample = True
    model.gradient_checkpointing_enable()

    if "lora" in config_dict:
        lora_config = LoraConfig(
            target_modules=find_all_linear_names(model),
            **config_dict["lora"],
        )
        print("Target modules for LoRA:", find_all_linear_names(model))
        model = get_peft_model(model, lora_config)

    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, tokenizer


class SaveLoRACallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        lora_save_path = os.path.join(checkpoint_dir, "lora")
        model = kwargs["model"]
        model.save_pretrained(lora_save_path)
        return control


class LogStepDataCallback(TrainerCallback):
    def __init__(self, dataset_indices, subset_info):
        self.dataset_indices = dataset_indices
        self.subset_info = subset_info
        self.step = 0

    def on_step_end(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        if self.step >= len(self.dataset_indices):
            return control
        index_path = os.path.join(checkpoint_dir, "data_indices.txt")
        with open(index_path, "w") as f:
            f.write(f"Subset: {self.subset_info}\n")
            f.write(f"Step: {state.global_step}\n")
            f.write(f"Indices: {self.dataset_indices[self.step]}\n")
        self.step += 1
        return control


def tokenize_dataset(dataset, tokenizer, dataset_name):
    col_names = dataset.column_names
    def tokenize_fn(batch, indices=None):
        batch['index'] = indices
        if dataset_name == "WaterDrum-Ax":
            texts = batch["text"]
        else:
            texts = [
                f"<s>[INST] {q.strip()} [/INST] {a.strip()}</s>"
                for q, a in zip(batch["question"], batch["answer_split"])
            ]
        
        encoded = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=config.max_seq_length,
        )
        # now add the index field into the return dict
        encoded["index"] = indices
        return encoded
        # return tokenizer(
        #     texts,
        #     truncation=True,
        #     padding="max_length",
        #     max_length=config.max_seq_length,
        # )
        
    return dataset.map(
        tokenize_fn,
        with_indices=True,
        batched=True,
        remove_columns=col_names,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["WaterDrum-TOFU", "WaterDrum-Ax"],
        required=True,
    )
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--split", type=str, default="forget")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    utils.set_seed(args.seed)

    with open(args.config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = type("Config", (object,), config_dict)()

    if args.dataset_name == "WaterDrum-TOFU":
        config.pretrained_model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
    else:
        config.pretrained_model_name_or_path = "meta-llama/Llama-2-7b-hf"

    model, tokenizer = load_model_and_tokenizer(config, config_dict)

    dataset = load_dataset(
        f"Glow-AI/{args.dataset_name}",
        args.subset,
        split=args.split
    )
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, args.dataset_name)

    output_dir = os.path.join("checkpoints", f"{args.dataset_name}-{args.subset}-{args.split}")
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = os.path.join(output_dir, "logs")

    batch_size = config.train_batch_size
    accum_steps = config.gradient_accumulation_steps
    steps_per_epoch = ceil(len(tokenized_dataset) / (batch_size * accum_steps))

    dataset_indices = []
    for _ in range(config.num_epochs):
        for i in range(steps_per_epoch):
            start_idx = i * batch_size * accum_steps
            end_idx = min(start_idx + batch_size * accum_steps, len(tokenized_dataset))
            dataset_indices.append(list(range(start_idx, end_idx)))

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,
        learning_rate=config.learning_rate,
        logging_dir=logging_dir,
        logging_steps=max(1, (steps_per_epoch * config.num_epochs) // 20),
        save_steps=1, # change this when doing full set
        save_strategy="steps",
        bf16=True,
        bf16_full_eval=True,
        seed=args.seed,
        optim='paged_adamw_32bit'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[
            SaveLoRACallback(),
            LogStepDataCallback(dataset_indices, subset_info=f"{args.dataset_name}-{args.subset}-{args.split}"),
        ],
    )

    trainer.train()
