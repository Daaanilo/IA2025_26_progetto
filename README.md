# HeRoN in Crafter: Adaptive Decision Making NPC

This project extends the HeRoN (Helper-Reviewer-NPC) architecture to the Crafter environment, a simplified version of Minecraft designed for Reinforcement Learning research.

## 🎯 Project Overview

**HeRoN Architecture** consists of three components:
- **NPC**: A Reinforcement Learning agent (Deep Q-Network) that plays in Crafter
- **Helper**: A zero-shot LLM that suggests sequences of coherent actions
- **Reviewer**: A fine-tuned LLM that evaluates Helper's suggestions and provides feedback

**Crafter Environment**: An open-world survival game with 22 unlockable achievements where the agent must:
- Find food and water
- Build shelter and sleep
- Defend against monsters
- Gather materials and craft tools

## 📁 Project Structure

```
IA2025_26_progetto/
├── src/
│   ├── environment/       # Crafter environment wrapper
│   ├── agents/           # DQN NPC implementation
│   └── llm/              # Helper and Reviewer LLMs
├── configs/              # Configuration files
├── data/                 # Training datasets
├── models/               # Saved models
├── notebooks/            # Jupyter notebooks for analysis
├── scripts/              # Training and evaluation scripts
└── tests/                # Unit tests
```

## 🚀 Setup

### 1. Create Conda Environment

#### Option A: Using Conda (Recommended)

```powershell
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate ia2025
```

#### Option B: Using Micromamba (Alternative)

```powershell
# Create environment from environment.yml
micromamba create -f environment.yml

# Activate the environment
micromamba activate ia2025
```

### 2. Install Crafter Separately

```powershell
# After activating the environment
pip install git+https://github.com/danijar/crafter.git
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_openai_api_key_here
WANDB_API_KEY=your_wandb_key_here  # Optional, for experiment tracking
```

## 📚 Usage

### Training the NPC

```powershell
# Make sure the ia2025 environment is activated
conda activate ia2025

python scripts/train_npc.py --config configs/dqn_config.yaml
```

### Fine-tuning the Reviewer

```powershell
# Make sure the ia2025 environment is activated
conda activate ia2025

python scripts/finetune_reviewer.py --config configs/reviewer_config.yaml
```

### Evaluating HeRoN

```powershell
# Make sure the ia2025 environment is activated
conda activate ia2025

python scripts/evaluate_heron.py --episodes 100
```

## 🎓 Objectives

1. **Fine-tune Reviewer**: Adapt the Reviewer to Crafter tasks
2. **Modify Helper**: Generate sequences of actions instead of single actions
3. **Implement NPC**: Develop DQN agent for Crafter
4. **Evaluate Performance**: Assess HeRoN's performance on Crafter's 22 achievements

## 📖 References

- **HeRoN Paper**: "HeRoN: A Multi-Agent RL–LLM Framework for Adaptive NPC Behavior in Interactive Environments"
- **Crafter Paper**: "Benchmarking The Spectrum of Agent Capabilities"
- **HeRoN Repository**: https://github.com/Seldre99/HeRoN
- **Crafter Repository**: https://github.com/danijar/crafter

## 📊 Results

Results and analysis will be documented here after training and evaluation.

## 🤝 Contributing

This is an academic project for the AI course 2025/26.

## 📝 License

This project is for educational purposes only.
