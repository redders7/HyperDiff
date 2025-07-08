import argparse
import os
import torch
from peft import PeftConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(
        description="Inspect trainable LoRA adapter parameters"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True,
        help="Path to the checkpoint directory, e.g. ~/scripts/checkpoints/.../checkpoint-1"
    )
    args = parser.parse_args()

    # Construct the LoRA adapter path
    lora_path = os.path.join(args.checkpoint_dir, "lora")
    if not os.path.isdir(lora_path):
        print(f"ERROR: Directory not found: {lora_path}")
        return

    # Load PEFT config
    peft_config = PeftConfig.from_pretrained(lora_path)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        is_trainable=True, 
    )
   
    trainable = [
        (name, tuple(param.shape))
        for name, param in model.named_parameters() if param.requires_grad
    ]

    if not trainable:
        print("No trainable parameters found.")
    else:
        print(f"Found {len(trainable)} trainable parameters:")
        for name, shape in trainable:
            print(f" - {name}: {shape}")

if __name__ == "__main__":
    main()
