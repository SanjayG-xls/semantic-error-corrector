import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# 1. Configuration
# We are using a highly capable, small coding model.
model_id = "deepseek-ai/deepseek-coder-1.3b-instruct" 
dataset_path = "dataset.json"
output_dir = "./verilog-lora-model"

print("Loading dataset...")
data = load_dataset("json", data_files=dataset_path)

# 2. Tokenizer Setup
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Fix for models without a default pad token

# 3. Model Setup (4-Bit Quantization for local GPUs)
print("Loading model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto", # Automatically maps to your GPU
    trust_remote_code=True
)

# 4. LoRA Setup (Low-Rank Adaptation)
# Instead of training 1.3 billion parameters, we train a tiny subset.
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    target_modules=["q_proj", "v_proj"], # The attention layers we want to tweak
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 5. Formatting Function
# This combines the instruction, input, and output so the model learns the full conversation flow.
# NEW FIXED VERSION
def formatting_prompts_func(example):
    text = f"Instruction: {example['instruction']}\nInput:\n{example['input']}\nOutput:\n{example['output']}"
    return text

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2, # Keep this small so we don't run out of memory
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    max_steps=50, # For our small dataset, 50 steps is enough to learn the pattern
    optim="paged_adamw_8bit",
    fp16=True,
)

# 7. Start Training
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=training_args,
    peft_config=lora_config,
    formatting_func=formatting_prompts_func,
)

trainer.train()

# 8. Save the Adapter
print(f"Training complete! Saving adapter to {output_dir}")
trainer.model.save_pretrained(output_dir)