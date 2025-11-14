import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSeq2SeqLM, EarlyStoppingCallback

# Locate JSONL dataset (tries several common locations)
possible_paths = [
    r'D:\\Progetto_AI2\\HeRoN\\game_scenarios_dataset_crafter.jsonl',
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'game_scenarios_dataset_crafter.jsonl'),
    os.path.join(os.path.dirname(__file__), '..', 'game_scenarios_dataset_crafter.jsonl'),
    'game_scenarios_dataset_crafter.jsonl'
]

dataset_path = None
for p in possible_paths:
    if os.path.exists(p):
        dataset_path = p
        break

if dataset_path is None:
    raise FileNotFoundError("game_scenarios_dataset_crafter.jsonl not found. Please generate dataset or update the path.")

df = pd.read_json(dataset_path, lines=True)  # insert dataset (JSONL)

df['input'] = df['prompt'] + " " + df['response']

dataset = Dataset.from_pandas(df[['input', 'instructions']])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Force CUDA detection
print(f"Using device: {device}")  # Verify GPU usage

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base") # LLM for Reviewer fine tuning
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device) # LLM for Reviewer fine tuning

def processes_function(examples):
    inputs = [ex for ex in examples['input']]
    targets = [ex for ex in examples['instructions']]

    # Optimized max_length: Crafter prompts are ~150 tokens, instructions ~80 tokens
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length")

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenizer_dataset = dataset.map(processes_function, batched=True)

train_test_split = tokenizer_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

training_args = TrainingArguments(
    output_dir="reviewer_output",
    eval_strategy='epoch',
    save_strategy='epoch',
    
    # GPU-optimized batch sizes for GTX 1660 Ti (6GB VRAM)
    per_device_train_batch_size=8,   # Reduced from 16 to fit 6GB VRAM
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,    # Effective batch size = 64 (same quality)
    
    # Training duration: 5 epochs optimal for T5 fine-tuning on ~5000 samples
    num_train_epochs=5,  # Increased from 3 for better convergence
    
    # Learning rate with warmup scheduler
    learning_rate=3e-5,  # Reduced from 5e-5 (T5 best practice)
    warmup_ratio=0.1,    # 10% warmup steps for stable training
    lr_scheduler_type='cosine',  # Cosine decay for smooth convergence
    
    # Regularization
    weight_decay=0.01,
    max_grad_norm=1.0,  # Gradient clipping to prevent explosions
    
    # Checkpointing and logging
    save_total_limit=2,  # Keep only 2 best checkpoints to save disk space
    load_best_model_at_end=True,  # Load best checkpoint after training
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    
    logging_dir="reviewer_logs",
    logging_steps=50,  # Log every 50 steps (~every 12% of epoch with 5800 samples)
    logging_first_step=True,
    
    # NVIDIA GPU optimizations
    fp16=True,  # ✅ ENABLE MIXED PRECISION (2x speedup, 50% less VRAM)
    dataloader_num_workers=0,  # Windows with GPU: use 0 to avoid multiprocessing issues
    dataloader_pin_memory=True,  # Faster GPU transfer
    
    # Reproducibility
    seed=42
)

if __name__ == '__main__':
    # Verify CUDA availability before training
    if not torch.cuda.is_available():
        print("⚠️ WARNING: CUDA not available! Training will be MUCH slower on CPU.")
        print("Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,  # Fix FutureWarning (replaces tokenizer parameter)
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Stop if no improvement for 2 epochs
    )

    print(f"Training on {len(train_dataset)} samples, validating on {len(test_dataset)} samples")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Total training steps: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
    print(f"Estimated training time: ~15-25 min (GPU) / ~2-3 hours (CPU)")

    trainer.train()

    # Absolute paths following HeRoN conventions
    output_path = r"D:\Progetto AI\IA2025_26_progetto\reviewer\reviewer_retrained"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"✅ Model saved to: {output_path}")
