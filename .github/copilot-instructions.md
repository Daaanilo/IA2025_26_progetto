# HeRoN - Multi-Agent RL-LLM Framework for Crafter

## Project Overview
HeRoN combines reinforcement learning with LLM reasoning for adaptive game AI in survival crafting environments. Three agents collaborate:
- **DQNAgent**: PyTorch neural network (state=43-dim, actions=17) with Double DQN + Prioritized Replay
- **CrafterHelper**: Zero-shot LLM (via LM Studio) generates 3-5 action sequences
- **InstructorAgent**: Fine-tuned T5 provides strategic feedback to refine Helper suggestions

**Environment**: Crafter (22 achievements, sparse rewards +1 per unlock). State = `[inventory(16), pos(2), status(3), achievements(22)]` = 43 dims.

## Core Architecture

### Environment (`classes/crafter_environment.py`)
43-dimensional state: `[inventory(16), position(2), status(3), achievements(22)]`
- Wraps `crafter.Env()`, extracts semantic features from `info` dict (inventory, achievements, player_pos)
- **All 17 actions always valid** (no masking needed)
- Sparse native rewards: +1 per achievement unlock only
- **Critical**: Always use `_extract_state()` to convert RGB observation → numerical state vector
- Inventory items: health, food, drink, energy, sapling, wood, stone, coal, iron, diamond, wood_pickaxe, stone_pickaxe, iron_pickaxe, wood_sword, stone_sword, iron_sword

### Training Loop (`training/heron_training.py`)
Threshold-based LLM involvement (decays 1.0→0.0 over episodes):
```python
# Per-EPISODE threshold decay (not per-step!)
for episode in range(episodes):
    threshold = max(0, threshold - 0.01)  # Decay once per episode
    
    for step in range(episode_length):
        if random() > threshold and episode < 600:
            # LLM workflow: Helper → Reviewer → Refined sequence
            sequence = helper.generate_action_sequence(state, info)
            if reviewer_available:
                feedback = reviewer.generate_suggestion(...)
                sequence = helper.generate_action_sequence(...)  # Re-prompt
            executor.execute_next(sequence)
        else:
            action = agent.act(state)  # Pure DQN
```

**Re-planning triggers** (Strategy B - interrupts 3-5 action sequences):
- Achievement unlock → re-query LLM with new context
- Health < 30% → DQN fallback for immediate survival  
- Critical resources depleted → re-query for gathering

### DQN Agent (`classes/agent.py`)
**Double DQN** with Prioritized Replay:
- Architecture: `Linear(128) → Linear(128) → Linear(64) → Linear(actions)` (PyTorch)
- Target network updated every 100 steps
- Priority α=0.6, β=0.4→1.0 (importance sampling)
- Memory: 10k transitions
- **Device auto-detection**: CUDA > CPU (no MPS support in current version)

**Critical patterns**:
```python
# State must be torch tensor on correct device
state_tensor = torch.FloatTensor(state).to(self.device)
q_values = self.model(state_tensor)

# Model persistence: 4 files (.pth, _memory.pkl, _priorities.pkl, _epsilon.txt)
agent.save("models/crafter_heron_final")  # Relative path
```

## Development Workflows

### Running Training
```powershell
conda activate HeRoN

# Main training script (F08 three-agent integration)
cd training
python heron_training.py

# DQN-only baseline
python DQN_training.py
```

**Prerequisites**:
1. LM Studio running on `http://127.0.0.1:1234` (load `llama-3.2-3b-instruct` or similar)
2. Reviewer model at `reviewer_retrained/` (optional, graceful fallback if missing)
3. Ensure `checkpoints/` and `models/` directories exist (auto-created)

### Baselines & Evaluation
```powershell
cd evaluation
python evaluation_report_generator.py # Compare HeRoN vs baselines
```

### Dataset Generation (F05)
```powershell
cd "dataset Reviewer"
python crafter_dataset_generation.py  # 50 episodes → ~2500 samples
```
Generates `game_scenarios_dataset_crafter.jsonl` with:
- `prompt`: State description (inventory, achievements, goals)
- `response`: Helper's raw LLM output
- `instructions`: Hand-crafted strategic feedback (5 quality tiers)
- `quality_score`: 0-1 based on achievement progress

**Reviewer Fine-Tuning** (F06):
```powershell
cd reviewer_fine_tuning
python reviewer_fine_tuning.py  # 5 epochs, saves to reviewer_retrained/
```

## Project-Specific Patterns

### LLM Response Parsing
```python
# Multi-action sequences: "[move_right], [do], [move_left]"
llm_response = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL)
matches = re.findall(r'\[(.*?)\]', llm_response)
action_ids = [CrafterHelper.ACTION_ID_MAP.get(m.strip().lower()) for m in matches[:5]]

# Fuzzy matching for typos (13 mappings in crafter_helper.py)
typo_map = {"place_rock": "place_stone", "make_pickaxe": "make_wood_pickaxe", ...}
```

### Reward Shaping (`training/heron_training.py`)
Crafter's native rewards are sparse (+1 achievement only). HeRoN adds adaptive intrinsic bonuses:
```python
bonus = resource_collection(+0.1) + health_mgmt(+0.05) + tier_progress(+0.05) + tools(+0.02)
shaped_reward = native_reward + bonus_total

# Curriculum stage multipliers (F09 - not yet implemented):
# Early stage: 1.5× resource_collection
# Mid stage: 2× tool_usage, 1.5× tier_progression  
# Late stage: 2× tier_progression, 1.5× health_management
```
**Track separately**: `native_reward` and `shaped_reward` for ablation studies.

## Development Guidelines

### Critical Patterns
```python
# State must be PyTorch tensor on correct device
state_tensor = torch.FloatTensor(state).to(agent.device)
action = agent.act(state_tensor)

# Memory replay only when sufficient samples
if len(agent.memory) > batch_size:
    agent.replay(batch_size)

# F09 Critical: Epsilon MUST start at 1.0, not 0.0
agent = DQNAgent(state_size=43, action_size=17, epsilon=1.0)

# Threshold decay PER EPISODE (not per step!)
for episode in range(episodes):
    # ... episode loop
    if episode < threshold_episodes:
        threshold = max(0, threshold - decay_per_episode)
```

### Model Persistence
```python
# F09 Fix: Relative paths (auto-creates directories)
agent.save("models/crafter_heron_final")  # Creates .pth, _memory.pkl, _priorities.pkl, _epsilon.txt
agent.load("checkpoints/best_model_ep42_ach15")  # Loads all 4 files

# Performance-based checkpointing
if episode_achievements > best_count:
    path = os.path.join("checkpoints", f"best_model_ep{e}_ach{achievements}")
    agent.save(path)
```

### Output Files (F09)
```python
# Training outputs
models/crafter_heron_final.*              # Final model (4 files)
checkpoints/best_model_ep*_ach*.*         # Best checkpoints
heron_crafter_extended_metrics.csv        # Per-episode metrics
heron_crafter_evaluation.json             # Summary statistics
hyperparameter_history.csv                # lr/epsilon/threshold per episode (F09 - not yet implemented)
training_config.json                      # Full configuration snapshot (F09 - not yet implemented)

# Evaluation plots (6 PNG files)
evaluation_plots/rewards_trend.png
evaluation_plots/achievements_heatmap.png
evaluation_plots/efficiency_scatter.png
evaluation_plots/helper_dependency.png
evaluation_plots/convergence_analysis.png
evaluation_plots/multi_metric_dashboard.png
```

## Common Pitfalls

### 1. Epsilon Initialization (F09 Critical Fix)
```python
# ❌ WRONG - No exploration!
self.epsilon = 0.0

# ✅ CORRECT - Start with full exploration
agent = DQNAgent(state_size=43, action_size=17, epsilon=1.0)
```

### 2. Threshold Decay Per-Step (F08 Bug)
```python
# ❌ WRONG - Decays to 0 in first episode!
while not done:
    threshold = max(0, threshold - 0.01)  # Per-step decay

# ✅ CORRECT - Decay per episode
for episode in range(episodes):
    # ... episode loop
    threshold = max(0, threshold - 0.01)  # After episode completes
```

### 3. LM Studio Connection
```powershell
# Verify LM Studio is running
(Invoke-WebRequest -Uri "http://127.0.0.1:1234/v1/models").Content | ConvertFrom-Json

# If connection fails, start LM Studio server and load model
```

### 4. State Size Mismatch
```python
# Crafter: state_size=43, action_size=17 (current)
# Battle (deprecated): state_size=36, action_size=9
agent = DQNAgent(state_size=43, action_size=17)  # Always verify
```

### 5. Reviewer Model Path
```python
# Training continues without Reviewer if model not found
REVIEWER_MODEL_PATH = "reviewer_retrained"  # Generated by F06
# Run reviewer_fine_tuning.py first if model missing
```

## Feature Development Workflow

### Implementation Process
1. Check `features.md` for specifications
2. Mark completed (✓) in `features.md` with implementation notes
3. **Always update `modifiche.md`** after completion:
   - Feature ID (e.g., "F09: Iterative Training")
   - Implementation approach and key decisions
   - File diff summary and trade-offs

### Current Status
- ✅ **F01-F06**: Core architecture complete (Environment, DQN, Helper, Dataset, Reviewer)
- ✅ **F08**: Three-agent training loop
- ✅ **F09**: Curriculum learning + hyperparameter scheduling + iterative refinement
- ✅ **F10**: Evaluation system with baselines
- ⏳ **F11-F14**: Testing, benchmarking, analysis, documentation

### Key Files
- `classes/crafter_environment.py`: Environment wrapper (43-dim state extraction)
- `classes/agent.py`: Double DQN with prioritized replay
- `classes/crafter_helper.py`: LLM action sequence generator
- `classes/instructor_agent.py`: T5 Reviewer
- `training/heron_training.py`: Three-agent training (F08)
- `training/DQN_training.py`: DQN-only baseline
- `dataset Reviewer/crafter_dataset_generation.py`: Dataset generation for Reviewer
- `reviewer_fine_tuning/reviewer_fine_tuning.py`: T5 fine-tuning on Crafter data
- `evaluation/evaluation_system.py`: Metrics aggregation + convergence detection