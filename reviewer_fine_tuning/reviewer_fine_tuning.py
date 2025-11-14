import torch
import json
from datasets import Dataset
from transformers import TrainingArguments, Trainer, T5Tokenizer, T5ForConditionalGeneration
import os

# Load JSONL dataset from Crafter dataset generation (F05)
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'game_scenarios_dataset_crafter.jsonl')

print(f"Loading dataset from {DATASET_PATH}...")
data = []
if os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"✓ Loaded {len(data)} samples from JSONL")
else:
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run crafter_dataset_generation.py first.")

# Prepare data for fine-tuning: input = state + Helper response, target = strategic feedback
processed_data = {
    'input': [f"{item['prompt']} {item['response']}" for item in data],
    'instructions': [item['instructions'] for item in data]
}

dataset = Dataset.from_dict(processed_data)
print(f"Dataset prepared with {len(dataset)} samples")

# Device auto-detection: CUDA > CPU (no MPS support as per HeRoN architecture)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load T5 model for Reviewer fine-tuning (F06)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)
print(f"✓ Model loaded: google/flan-t5-base")

def processes_function(examples):
    inputs = [ex for ex in examples['input']]
    targets = [ex for ex in examples['instructions']]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenizer_dataset = dataset.map(processes_function, batched=True)

train_test_split = tokenizer_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Training configuration (F06 - Reviewer Fine-tuning)
OUTPUT_DIR = "reviewer_training_output"
LOGGING_DIR = os.path.join(OUTPUT_DIR, "logs")
REVIEWER_MODEL_PATH = "reviewer_retrained"  # Final model output

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,  # 5 epochs as per copilot-instructions
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=3,
    logging_dir=LOGGING_DIR,
    logging_steps=10,
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss'
)

print(f"Training configuration:")
print(f"  - Output dir: {OUTPUT_DIR}")
print(f"  - Epochs: {training_args.num_train_epochs}")
print(f"  - Batch size: {training_args.per_device_train_batch_size}")
print(f"  - Learning rate: {training_args.learning_rate}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

print("\nStarting training...")
trainer.train()

# Save final model to reviewer_retrained/ (used by InstructorAgent in F08)
print(f"\nSaving fine-tuned Reviewer to {REVIEWER_MODEL_PATH}/...")
model.save_pretrained(REVIEWER_MODEL_PATH)
tokenizer.save_pretrained(REVIEWER_MODEL_PATH)

print(f"✓ Training complete!")
print(f"✓ Reviewer model saved to {REVIEWER_MODEL_PATH}/")
print(f"\nModel is ready for use in heron_training.py (F08)")
print(f"  InstructorAgent will load from: {REVIEWER_MODEL_PATH}")
