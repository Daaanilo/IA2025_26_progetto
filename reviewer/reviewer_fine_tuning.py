import torch
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments, Trainer,AutoTokenizer, AutoModelForSeq2SeqLM

df = pd.read_csv('D:\\Progetto_AI2\\HeRoN\\game_scenarios_dataset_crafter.csv') # insert dataset

df['input'] = df['prompt'] + " " + df['response']

dataset = Dataset.from_pandas(df[['input', 'instructions']])

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base") # LLM for Reviewer fine tuning
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device) # LLM for Reviewer fine tuning

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

training_args = TrainingArguments(
    output_dir="reviewer_output",
    eval_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=3,
    logging_dir="reviewer_logs",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("reviewer_retrained") # Insert new Reviewer path
tokenizer.save_pretrained("reviewer_retrained") # Insert new Reviewer path
