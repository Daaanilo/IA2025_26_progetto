# HeRoN - Multi-Agent RL-LLM for Crafter

## Architecture Overview
Three-agent system combining reinforcement learning with LLM reasoning:
- **DQNAgent** (`classes/agent.py`): PyTorch Double DQN + Prioritized Replay Buffer (10k transitions, α=0.6, β=0.4→1.0)
- **CrafterHelper** (`classes/crafter_helper.py`): Zero-shot LLM (via LM Studio) generates 3-5 action sequences
- **InstructorAgent** (`classes/instructor_agent.py`): Fine-tuned T5 refines Helper suggestions

**State**: 43-dim vector = `[inventory(16), position(2), status(3), achievements(22)]` extracted from info dict  
**Actions**: 17 discrete (0=noop, 1-4=movement, 5=do, 6=sleep, 7-10=place_*, 11-16=make_*)  
**Rewards**: Sparse (+1/achievement) + shaped bonuses (resources +0.1, health +0.05, tier +0.05, tools +0.02)

## Project Structure
```
classes/               # Core agent implementations
├── agent.py          # Double DQN with prioritized replay
├── crafter_environment.py  # 43-dim state extraction wrapper
├── crafter_helper.py # LLM sequence generator + re-planning
└── instructor_agent.py     # T5 Reviewer

training/             # Training scripts
├── heron_training.py       # Full 3-agent system
└── DQN_training.py        # Baseline (no LLM)

evaluation/           # Metrics & visualization
├── evaluation_system.py   # Achievement tracking
├── evaluation_plots.py    # Learning curves
└── evaluation_report_generator.py

reviewer_fine_tuning/ # T5 Reviewer training
├── reviewer_fine_tuning.py
└── game_scenarios_dataset_crafter.jsonl

dataset Reviewer/     # Dataset generation for Reviewer
└── crafter_dataset_generation.py
```

## Critical Workflows

### 1. Training Loop Pattern
**Three-agent decision flow** in `training/heron_training.py`:
```python
# CRITICAL: Threshold decay PER EPISODE (not per step!)
threshold = 1.0  # Start with 100% LLM involvement
for episode in range(episodes):
    if episode < threshold_episodes:
        threshold = max(0, threshold - 0.01)  # Decays after each episode
    
    for step in range(max_steps):
        # Decision: LLM or DQN?
        if random() > threshold and episode < 600:
            # LLM workflow: Helper → Reviewer → Refined Helper
            sequence, response = helper.generate_action_sequence(state, info, previous_info)
            if use_reviewer and instructor:
                game_desc = helper.describe_crafter_state(state, info, previous_info)
                feedback = instructor.generate_suggestion(game_desc, response)
                # Re-prompt Helper with Reviewer feedback
                sequence = helper.generate_action_sequence(...)
        else:
            action = agent.act(state, env)  # Pure DQN fallback
```

**Re-planning triggers** (via `helper.should_replan()` - interrupts 3-5 action sequences):
- Achievement unlock → new LLM query with updated context
- Health ≤ 5 → immediate DQN fallback (critical survival)
- Health drops below 30% → re-query for health management
- Resource depletion → re-query when key resources reach 0

### 2. Setup & Execution
```powershell
# Prerequisites: Conda env + LM Studio server running
conda activate HeRoN

# Verify LM Studio is running (must return model list)
(Invoke-WebRequest -Uri "http://127.0.0.1:1234/v1/models").Content | ConvertFrom-Json

# Full HeRoN training (DQN + Helper + Reviewer)
cd training; python heron_training.py

# DQN-only baseline (no LLM)
python DQN_training.py

# Generate Reviewer training data (50 episodes → ~2500 samples)
cd "dataset Reviewer"; python crafter_dataset_generation.py

# Fine-tune Reviewer T5 model (5 epochs, saves to reviewer_retrained/)
cd reviewer_fine_tuning; python reviewer_fine_tuning.py

# Compare HeRoN vs baselines with plots
cd evaluation; python evaluation_report_generator.py
```

### 3. Testing & Debugging
```powershell
# Test environment wrapper (43-dim state extraction)
python test_crafter_env.py

# Test LM Studio connectivity
python test_lmstudio_connection.py
```

## Critical Patterns

### 1. State Handling (Environment Wrapper)
`CrafterEnv` extracts 43-dim state from Crafter's info dict - **DO NOT use raw RGB observations**:
```python
# ✅ CORRECT - Returns 43-dim vector from info dict
state = env.reset()  # Returns [16 inventory + 2 pos + 3 status + 22 achievements]
next_state, reward, done, info = env.step(action)

# State components (see crafter_environment.py):
# - inventory[0:16]: health, food, drink, energy, sapling, wood, stone, coal, iron, 
#                    diamond, wood_pickaxe, stone_pickaxe, iron_pickaxe, 
#                    wood_sword, stone_sword, iron_sword
# - position[16:18]: player_x, player_y (normalized to [0,1])
# - status[18:21]: discount (alive/dead), sleeping, daylight
# - achievements[21:43]: 22 one-hot achievement flags

# DQN requires PyTorch tensor on correct device
state_tensor = torch.FloatTensor(state).to(agent.device)  # Auto-detects CUDA/CPU
q_values = agent.model(state_tensor)
```

### 2. Model Persistence (4-File System)
DQNAgent saves/loads **4 files** automatically:
```python
# Saves: .pth (weights), _target.pth (Double DQN), _memory.pkl, _priorities.pkl, _epsilon.txt
agent.save("models/crafter_heron_final")  # Auto-creates dirs

# Load all 4 files
agent.load("checkpoints/best_model_ep42_ach15")

# Performance-based checkpointing pattern (training/heron_training.py):
if episode_achievements > best_achievement_count:
    best_achievement_count = episode_achievements
    agent.save(f"checkpoints/best_model_ep{e}_ach{episode_achievements}")
    print(f"[Checkpoint] New best: {episode_achievements} achievements")
```

### 3. LLM Response Parsing & Typo Correction
`CrafterHelper` expects bracketed actions, handles typos via fuzzy matching:
```python
# Expected LLM format: "[move_right], [do], [move_left]" (3-5 actions max)
llm_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)
action_ids = [CrafterHelper.ACTION_ID_MAP.get(m.strip().lower()) 
              for m in re.findall(r'\[(.*?)\]', llm_response)[:5]]

# Typo correction map (13 common errors in crafter_helper.py):
TYPO_MAP = {
    'place_rock': 'place_stone',      # Most common mistake
    'make_pickaxe': 'make_wood_pickaxe',
    'wood pickaxe': 'make_wood_pickaxe',
    'collect_wood': None,             # Invalid - use [do] instead
    'gather': None,                   # Invalid - use [do] instead
    'mine': None                      # Invalid - use [do] instead
}
```

### 4. Action ID Mapping (Official Crafter Order)
**MUST use exact IDs** - common mistakes annotated:
```python
ACTION_NAMES = {
    0: 'noop',               # ✓ Valid - wait/idle
    1: 'move_left',  2: 'move_right',  3: 'move_up',  4: 'move_down',
    5: 'do',                 # ✓ Multi-purpose: collect wood/stone, attack, etc.
    6: 'sleep',              # ✓ Restore health/energy
    7: 'place_stone',        # ❌ NOT 'place_rock'
    8: 'place_table',        # ✓ Required before crafting tools
    9: 'place_furnace',      # ✓ Required for smelting iron
    10: 'place_plant',       # ✓ Grow resources
    11: 'make_wood_pickaxe', # ❌ NOT 'make_pickaxe' or 'craft_pickaxe'
    12: 'make_stone_pickaxe', 13: 'make_iron_pickaxe',
    14: 'make_wood_sword', 15: 'make_stone_sword', 16: 'make_iron_sword'
}
# Total: 17 actions (NO action 17+)
```

### 5. Reward Shaping System
Track **native_reward** (sparse +1/achievement) and **shaped_reward** separately:
```python
# In heron_training.py - CrafterRewardShaper class:
shaped_reward, bonus_components = reward_shaper.calculate_shaped_reward(
    native_reward, next_state, info, previous_info, action
)

# Bonus components:
# - resource_collection: +0.1 per resource collected via [do]
# - health_management: +0.05 for eating/drinking/health increase
# - tier_progression: +0.05 for advancing through achievement chains
# - tool_usage: +0.02 for crafting tools

# DQN trains on shaped_reward, but log both for ablation studies
total_native_reward += native_reward
total_shaped_reward += shaped_reward
```

## Common Pitfalls (Fix Immediately)

### 1. Epsilon Initialization
```python
# ❌ WRONG - No exploration!
agent = DQNAgent(epsilon=0.0)

# ✅ CORRECT - Start with full exploration
agent = DQNAgent(state_size=43, action_size=17, epsilon=1.0)  # Decays to 0.05
```

### 2. Threshold Decay Timing
```python
# ❌ WRONG - Decays to 0 in first episode (100 steps × 0.01 = game over!)
while not done:
    threshold -= 0.01  # Inside step loop

# ✅ CORRECT - Decay per episode (100 episodes to decay 1.0→0.0)
for episode in range(episodes):
    # ... run entire episode here ...
    
    # AFTER episode completes:
    if episode < threshold_episodes:
        threshold = max(0, threshold - 0.01)
```

### 3. LM Studio Connectivity
```powershell
# Verify server is running and model loaded
(Invoke-WebRequest -Uri "http://127.0.0.1:1234/v1/models").Content | ConvertFrom-Json
# Expected: {"data": [{"id": "llama-3.2-3b-instruct", ...}]}

# If connection fails:
# 1. Start LM Studio application
# 2. Load model (llama-3.2-3b-instruct recommended)
# 3. Start local server on port 1234
```

### 4. State Dimensions Mismatch
```python
# ❌ WRONG - Using old Battle environment dimensions
agent = DQNAgent(state_size=36, action_size=9)

# ✅ CORRECT - Crafter environment dimensions
assert env.state_size == 43 and env.action_size == 17
agent = DQNAgent(state_size=43, action_size=17)
```

### 5. Reviewer Model Availability
```python
# Training continues gracefully if reviewer_retrained/ missing
# To enable full three-agent workflow:
# 1. Run dataset generation: cd "dataset Reviewer"; python crafter_dataset_generation.py
# 2. Run fine-tuning: cd reviewer_fine_tuning; python reviewer_fine_tuning.py
# This creates reviewer_retrained/ directory with T5 model + tokenizer
```

### 6. Action Sequence Execution
```python
# ❌ WRONG - Infinite loop if sequence never exhausts
while executor.current_sequence:
    action = executor.current_sequence[executor.current_sequence_index]

# ✅ CORRECT - Check index bounds
if executor.current_sequence and executor.current_sequence_index < len(executor.current_sequence):
    action = executor.current_sequence[executor.current_sequence_index]
    executor.current_sequence_index += 1
else:
    # Sequence exhausted - get new one or use DQN
    action = agent.act(state, env)
```

## File Organization

### Training Outputs
```
models/crafter_heron_final.*              # Final model (4 files)
checkpoints/best_model_ep*_ach*.*         # Performance-based snapshots
heron_crafter_extended_metrics.csv        # Per-episode metrics
heron_crafter_evaluation.json             # Summary stats
evaluation_plots/*.png                    # 6 visualization plots
```

### Core Files
- `classes/crafter_environment.py`: 43-dim state extraction from Crafter info dict
- `classes/agent.py`: Double DQN + prioritized replay (PyTorch, CUDA/CPU)
- `classes/crafter_helper.py`: LLM sequence generator + re-planning logic
- `classes/instructor_agent.py`: T5 Reviewer (fine-tuned on ~2500 samples)
- `training/heron_training.py`: Three-agent training loop with threshold decay
- `training/DQN_training.py`: DQN-only baseline for ablation studies
- `evaluation/evaluation_system.py`: Metrics aggregation + convergence detection

### Dependencies
```bash
# Core ML/RL
pip install torch torchvision transformers datasets accelerate
pip install numpy pandas matplotlib seaborn tqdm scikit-learn

# LLM integration
pip install lmstudio  # Requires LM Studio server running

# Crafter environment (CRITICAL on Windows)
$env:PYTHONUTF8="1"; pip install crafter>=1.8.3

# Test installations
python test_crafter_env.py
python test_lmstudio_connection.py
```

## Development Notes

### Reward Shaping for Ablation Studies
Track `native_reward` (sparse +1/achievement) and `shaped_reward` separately:
```python
# In training loops - always track both for comparison
total_native_reward += native_reward
shaped_reward = native_reward + resource_bonus + health_bonus + tier_bonus + tool_bonus
total_shaped_reward += shaped_reward

# DQN trains on shaped_reward
agent.remember(state, action, shaped_reward, next_state, done)

# Log both in metrics for plotting baseline comparisons
metrics.append({
    'episode': e,
    'native_reward': total_native_reward,
    'shaped_reward': total_shaped_reward,
    'shaped_bonus': total_shaped_reward - total_native_reward
})
```

### GPU/CPU Compatibility
```python
# Auto-detection pattern used throughout codebase
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# IMPORTANT: NO MPS (Apple Silicon) support in this project
# Crafter environment + PyTorch DQN only support CUDA/CPU
```

### Achievement Tracking Pattern
```python
# Track by NAME first, then convert to IDs
prev_achievements = set(k for k, v in previous_info.get('achievements', {}).items() if v >= 1)
curr_achievements = set(k for k, v in info.get('achievements', {}).items() if v >= 1)
newly_unlocked_names = curr_achievements - prev_achievements

# Convert names to IDs for evaluation system (22 official achievements)
from evaluation.evaluation_system import ACHIEVEMENT_NAME_TO_ID
newly_unlocked_ids = {ACHIEVEMENT_NAME_TO_ID[name] for name in newly_unlocked_names 
                     if name in ACHIEVEMENT_NAME_TO_ID}
evaluation_system.add_episode_achievements(episode, newly_unlocked_ids, current_step)
```

### Dataset Generation for Reviewer
```python
# Generate ~2500 training samples from 50 episodes
# Each Helper call → outcome evaluation → strategic feedback
cd "dataset Reviewer"
python crafter_dataset_generation.py  # Outputs game_scenarios_dataset_crafter.jsonl

# Fine-tune T5 model (5 epochs)
cd ../reviewer_fine_tuning
python reviewer_fine_tuning.py  # Outputs reviewer_retrained/ (model + tokenizer)

# Use in training
# heron_training.py automatically loads from reviewer_retrained/ if available
```

### DON'T GENERATE DOCUMENT CHAGNES 