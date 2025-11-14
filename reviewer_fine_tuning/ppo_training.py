import re
import torch
import json
from datasets import Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from transformers import AutoTokenizer
from tqdm import tqdm
import os


def processes_function(examples):
    inputs = [ex for ex in examples['input']]
    targets = [ex for ex in examples['instructions']]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def calculate_reward(ideal_feedback, suggested_feedback):
    """
    Calculate reward based on quality of Reviewer's strategic feedback.
    
    Args:
        ideal_feedback: Target strategic feedback (from dataset)
        suggested_feedback: Model-generated feedback
    
    Returns:
        float: Reward score
    """
    # Positive reward for generating any feedback
    if not suggested_feedback or len(suggested_feedback.strip()) < 10:
        return -5.0  # Penalty for too short or empty feedback
    
    # Check for key strategic terms (Crafter-specific)
    strategic_terms = [
        'achievement', 'resource', 'collect', 'craft', 'health', 
        'wood', 'stone', 'iron', 'pickaxe', 'sword', 'table',
        'prioritize', 'efficiency', 'progression', 'tier'
    ]
    
    term_count = sum(1 for term in strategic_terms if term.lower() in suggested_feedback.lower())
    
    # Base reward for term coverage
    reward = term_count * 0.5
    
    # Check for action suggestions (bracketed actions like [do], [move_right])
    action_pattern = r'\[([^\]]+)\]'
    ideal_actions = set(re.findall(action_pattern, ideal_feedback.lower()))
    suggested_actions = set(re.findall(action_pattern, suggested_feedback.lower()))
    
    # Reward for matching actions
    if ideal_actions and suggested_actions:
        action_overlap = len(ideal_actions & suggested_actions) / max(len(ideal_actions), 1)
        reward += action_overlap * 3.0
    
    # Bonus for quality indicators
    quality_indicators = ['EXCELLENT', 'GOOD', 'CRITICAL', 'WARNING', 'SUGGESTION']
    has_quality_feedback = any(ind in suggested_feedback for ind in quality_indicators)
    if has_quality_feedback:
        reward += 2.0
    
    # Length penalty (too verbose is bad)
    if len(suggested_feedback) > 500:
        reward -= 1.0
    
    return reward


def collators(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# Device auto-detection: CUDA > CPU (no MPS support)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load fine-tuned Reviewer model from F06
REVIEWER_MODEL_PATH = "reviewer_retrained"
print(f"Loading Reviewer model from {REVIEWER_MODEL_PATH}...")

if not os.path.exists(REVIEWER_MODEL_PATH):
    raise FileNotFoundError(
        f"Reviewer model not found at {REVIEWER_MODEL_PATH}. "
        f"Run reviewer_fine_tuning.py first (F06)."
    )

tokenizer = AutoTokenizer.from_pretrained(REVIEWER_MODEL_PATH)
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(REVIEWER_MODEL_PATH).to(device)
print(f"✓ Model loaded from {REVIEWER_MODEL_PATH}")

# Load JSONL dataset
DATASET_PATH = 'game_scenarios_dataset_crafter.jsonl'
print(f"Loading dataset from {DATASET_PATH}...")

data = []
if os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"✓ Loaded {len(data)} samples from JSONL")
else:
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run crafter_dataset_generation.py first.")

# Prepare data for PPO training
processed_data = {
    'input': [f"{item['prompt']} {item['response']}" for item in data],
    'instructions': [item['instructions'] for item in data]
}

dataset = Dataset.from_dict(processed_data)
tokenizer_dataset = dataset.map(processes_function, batched=True)

ppo_config = PPOConfig(
    learning_rate=5e-7,
    ppo_epochs=1,
    mini_batch_size=1,
    batch_size=1
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
)

generation_kwards = {
    "temperature": 0.4,
    "top_k": 50,
    "top_p": 0.8,
    "max_new_tokens": 128,  # Limit output length
}


def train_ppo(epochs):
    """Train Reviewer with PPO reinforcement learning."""
    print(f"\nStarting PPO training for {epochs} epochs...")
    
    for i in tqdm(range(epochs), desc="PPO Epochs"):
        epoch_rewards = []
        
        for batch in dataset:
            input_tensor = []
            response_tensor = []
            reward_tensor = []
            game_input = batch['input']
            target_feedback = batch['instructions']

            inputs = tokenizer(game_input, return_tensors="pt").to(device).input_ids
            input_tensor.append(inputs[0])

            response = ppo_trainer.generate(inputs[0], **generation_kwards)
            response_tensor.append(response[0])

            response_text = tokenizer.decode(response[0], skip_special_tokens=True)

            reward = calculate_reward(target_feedback, response_text)
            reward_tensor.append(torch.tensor(reward, dtype=torch.float))
            epoch_rewards.append(reward)

            stats = ppo_trainer.step(input_tensor, response_tensor, reward_tensor)
        
        # Epoch summary
        avg_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        print(f"\nEpoch {i+1}/{epochs} complete")
        print(f"  Average reward: {avg_reward:.3f}")
        print(f"  KL divergence: {stats.get('objective/kl', 'N/A')}")
        print(f"  Returns mean: {stats.get('ppo/returns/mean', 'N/A')}")

    # Save PPO-trained model
    output_path = "reviewer_retrained_ppo"
    print(f"\nSaving PPO-trained model to {output_path}/...")
    ppo_trainer.save_pretrained(output_path)
    print(f"✓ PPO training complete! Model saved to {output_path}/")

train_ppo(3)

