# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HeRoN in Crafter is a multi-agent reinforcement learning system for the Crafter environment (Minecraft-inspired survival game). It combines three agents:
- **DQNAgent**: Double DQN with Prioritized Experience Replay
- **CrafterHelper**: Zero-shot LLM (via LM Studio) that suggests action sequences
- **InstructorAgent**: Fine-tuned T5 model that refines Helper suggestions

**Authors**: Danilo Gisolfi & Vincenzo Maiellaro (Universita degli Studi di Salerno)

## Commands

```bash
# Install dependencies (Windows requires UTF-8 for Crafter)
pip install -r requirements.txt
$env:PYTHONUTF8="1"; pip install crafter>=1.8.3

# Verify setup
python test_crafter_env.py
python test_lmstudio_connection.py

# Training (from project root or training/ directory)
python training/DQN_training.py           # Baseline DQN only
python training/dqn_helper_training.py    # DQN + Helper
python training/HERON_final.py            # Full HeRoN (recommended)
python training/HERON_initial.py          # Fixed 100-step LLM assistance
python training/HERON_random.py           # Random LLM activation

# Reviewer fine-tuning
cd "dataset Reviewer" && python crafter_dataset_generation.py
cd reviewer_fine_tuning && python reviewer_fine_tuning.py

# Evaluation
cd evaluation && python evaluation_report_generator.py
```

## Architecture

### State Representation (43 dimensions)
Extracted by `CrafterEnv` from Crafter's info dict:
- **Inventory** (16): health, food, drink, energy, sapling, wood, stone, coal, iron, diamond, wood_pickaxe, stone_pickaxe, iron_pickaxe, wood_sword, stone_sword, iron_sword
- **Position** (2): x, y normalized to [0,1]
- **Status** (3): discount (alive/dead), sleeping, daylight
- **Achievements** (22): one-hot flags for all achievements

### Action Space (17 discrete)
```
0: noop, 1-4: move_left/right/up/down, 5: do, 6: sleep,
7-10: place_stone/table/furnace/plant,
11-16: make_wood/stone/iron_pickaxe, make_wood/stone/iron_sword
```

### DQN Network
```python
# classes/agent.py - DQNNetwork
Linear(43, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 17)
```

Key parameters:
- Memory: 5000 transitions (Prioritized Replay with alpha=0.6, beta=0.4->1.0)
- Gamma: 0.99, Target network update every 100 steps
- Epsilon: 1.0 -> 0.05 linear decay over 300 episodes

## Critical Patterns

### Threshold Decay (HERON_final.py)
LLM involvement decreases per STEP within each episode:
```python
KAPPA = 0.01  # threshold decreases by 0.01 per step
threshold = 1.0  # reset to 1.0 at start of each episode
for step in range(episode_length):
    use_llm = (random() > threshold) and (episode < threshold_episodes)
    threshold = max(0, threshold - KAPPA)  # after 100 steps, threshold=0
```

### Model Persistence
DQNAgent saves/loads 5 files automatically:
- `.pth` - model weights
- `_target.pth` - target network weights
- `_memory.pkl` - replay buffer
- `_priorities.pkl` - priority values
- `_epsilon.txt` - current exploration rate

```python
agent.save("checkpoints/model_ep50")  # creates 5 files
agent.load("checkpoints/model_ep50")  # loads all 5
```

### LLM Response Format
Helper expects bracketed actions: `[move_right], [do], [sleep]`

Common typo corrections in `crafter_helper.py`:
- `place_rock` -> `place_stone`
- `wood pickaxe` -> `make_wood_pickaxe`
- `collect_wood`, `gather`, `mine` -> invalid (use `[do]`)

### Re-planning Triggers (CrafterHelper.should_replan)
- Health <= 5: Critical - immediate action needed
- Health drops below 30% (6 HP)
- New achievement unlocked
- Key resource depleted (wood/stone/iron/coal/diamond reaches 0)

## File Structure

```
classes/
  agent.py              # DQNAgent with Double DQN + Prioritized Replay
  crafter_environment.py # CrafterEnv wrapper (43-dim state extraction)
  crafter_helper.py     # LLM action sequence generator + re-planning
  instructor_agent.py   # T5 Reviewer wrapper

training/
  HERON_final.py        # Full 3-agent system with threshold decay (recommended)
  HERON_initial.py      # Fixed 100-step LLM assistance variant
  HERON_random.py       # Random LLM activation variant
  DQN_training.py       # Pure DQN baseline
  dqn_helper_training.py # DQN + Helper (no Reviewer)
  reward_shaper.py      # Shaped reward calculation

evaluation/
  evaluation_system.py         # Achievement tracking + metrics
  evaluation_plots.py          # Visualization utilities
  evaluation_report_generator.py

reviewer_fine_tuning/
  reviewer_fine_tuning.py      # T5 supervised fine-tuning

"dataset Reviewer"/
  crafter_dataset_generation.py # Generate training data for Reviewer
```

## Prerequisites

- **LM Studio**: Must be running on port 1234 with a model loaded (e.g., qwen3-4b-2507)
- **GPU**: Auto-detects CUDA; falls back to CPU. No MPS support.
- **Reviewer model**: Optional - training proceeds without it if `reviewer_retrained_ppo/` doesn't exist

## Training Outputs

Each variant outputs to its directory (e.g., `training/heron_final_output/`):
- `*_metrics.jsonl` - per-episode metrics
- `*_achievement_statistics.json` - achievement unlock stats
- `*_evaluation.json` - summary statistics
- `checkpoints/` - best models saved by achievement count
- `plots/` - learning curve visualizations
