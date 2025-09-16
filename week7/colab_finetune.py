"""
Fine-tune LLaMA 3 8B with QLoRA using Unsloth on Google Colab

Instructions:
1. Open Google Colab (colab.research.google.com)
2. Change runtime to GPU (Runtime > Change runtime type > T4 GPU)
3. Upload your synthetic_qa_comprehensive.jsonl file to Colab
4. Run this script cell by cell
"""

# CELL 1: Install dependencies
get_ipython().system('pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
get_ipython().system('pip install --no-deps "xformers<0.0.27" trl peft accelerate bitsandbytes')

# CELL 2: Import and setup
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import json

# Check GPU
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")

# CELL 3: Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",  # Base LLaMA 3 8B
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Show trainable parameters
model.print_trainable_parameters()

# CELL 4: Load dataset
# Upload your synthetic_qa_comprehensive.jsonl file to Colab first
dataset = load_dataset("json", data_files="synthetic_qa_comprehensive.jsonl", split="train")
print(f"Loaded {len(dataset)} training examples")

# CELL 5: Setup training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # Adjust based on your needs
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
    ),
)

# CELL 6: Train
trainer.train()

# CELL 7: Save model
model.save_pretrained("llama3-academic-qa")
tokenizer.save_pretrained("llama3-academic-qa")

print("Training complete! Model saved to llama3-academic-qa/")

# CELL 8: Test the model
FastLanguageModel.for_inference(model)

# Test question
inputs = tokenizer(
    [
        "<|system|>You are a helpful academic Q&A assistant specialized in scholarly content.<|user|>What is the main contribution of recent research in computer vision?<|assistant|>"
    ], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
print(tokenizer.batch_decode(outputs)[0])