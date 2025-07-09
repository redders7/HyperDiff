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
from rouge_score import rouge_scorer
import json
import random

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
    
class EvaluateROUGECallback(TrainerCallback):
    def __init__(self, eval_dataset_raw, tokenizer, output_dir):
        self.eval_dataset_raw = eval_dataset_raw
        self.tokenizer = tokenizer
        self.output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        rouge_results = compute_rouge(model, self.tokenizer, self.eval_dataset_raw)

        results_dir = os.path.join(self.output_dir, "rouge_results")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, f"rouge_epoch_{int(state.epoch)}.json")
        with open(output_path, "w") as f:
            json.dump(rouge_results, f, indent=2)

        print(f"📊 ROUGE results saved to {output_path}")
        print("Epoch:", int(state.epoch))
        average_rouge1_recall = sum(rouge_results['rouge1_recall'].values()) / len(rouge_results['rouge1_recall'])
        average_rougeL_recall = sum(rouge_results['rougeL_recall'].values()) / len(rouge_results['rougeL_recall'])
        print("Average ROUGE-1 Recall:", average_rouge1_recall)
        print("Average ROUGE-L Recall:", average_rougeL_recall)
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

def compute_rouge(model, tokenizer, dataset):
    model.eval()
    input_texts = [item['question'] for item in dataset]
    ground_truths = [item['answer_split'] for item in dataset]
    indices = list(range(len(dataset)))

    gen_outputs = []
    for input_text in input_texts:
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
        output_ids = model.generate(input_ids, max_new_tokens=50)
        gen_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
        gen_outputs.append(gen_text.strip())

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_recall = {}
    rougeL_recall = {}
    for gen, gt, idx in zip(gen_outputs, ground_truths, indices):
        scores = scorer.score(gt, gen)
        rouge1_recall[idx] = scores['rouge1'].recall
        rougeL_recall[idx] = scores['rougeL'].recall

    return {'rouge1_recall': rouge1_recall, 'rougeL_recall': rougeL_recall}


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
    parser.add_argument("--dataset_size", type=int, default=-1)
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

    # If dataset_size is specified, sample a subset of the dataset
    # Else, use the full dataset
    if args.dataset_size > 0:
        random.seed(args.seed)
        all_indices = list(range(len(dataset)))
        random.shuffle(all_indices)
        sampled_indices = all_indices[:args.dataset_size]
        dataset = dataset.select(sampled_indices)

    # Split the dataset into 80% train and 20% eval (adjust as needed)
    split_datasets = dataset.train_test_split(test_size=0.2, seed=args.seed)
    train_dataset_raw = split_datasets['train']
    eval_dataset_raw = split_datasets['test']

    tokenized_dataset = tokenize_dataset(train_dataset_raw, tokenizer, args.dataset_name)
    tokenized_eval_dataset = tokenize_dataset(eval_dataset_raw, tokenizer, args.dataset_name)

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
            EvaluateROUGECallback(eval_dataset_raw, tokenizer, output_dir),
        ],
    )

    trainer.train()