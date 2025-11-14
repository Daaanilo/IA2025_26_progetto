# HeRoN - Multi-Agent RL-LLM for Crafter

## Architecture Overview
Three-agent system combining reinforcement learning with LLM reasoning:
- **DQNAgent** (`classes/agent.py`): PyTorch Double DQN + Prioritized Replay Buffer (10k transitions, α=0.6, β=0.4→1.0)
- **CrafterHelper** (`classes/crafter_helper.py`): Zero-shot LLM (via LM Studio) generates 3-5 action sequences
- **InstructorAgent** (`classes/instructor_agent.py`): Fine-tuned T5 refines Helper suggestions

**State**: 43-dim vector = `[inventory(16), position(2), status(3), achievements(22)]`  
**Actions**: 17 discrete (movement, do, sleep, place_*, make_*, noop)  
**Rewards**: Sparse (+1/achievement) + shaped bonuses (resources +0.1, health +0.05, tier +0.05, tools +0.02)

## Critical Workflows

### Training Loop Pattern (`training/heron_training.py`)
```python
# CRITICAL: Threshold decay PER EPISODE (not per step!)
for episode in range(episodes):
    threshold = max(0, threshold - 0.01)  # Decays 1.0→0.0 over 100 episodes
    
    for step in range(max_steps):
        if random() > threshold and episode < 600:
            # LLM workflow: Helper → Reviewer → Refined sequence
            sequence = helper.generate_action_sequence(state, info)
            if reviewer_available:
                feedback = reviewer.generate_suggestion(...)
                sequence = helper.generate_action_sequence(...)  # Re-prompt with feedback
        else:
            action = agent.act(state, env)  # Pure DQN
```

**Re-planning triggers** (interrupts 3-5 action sequences):
- Achievement unlock → new LLM query with updated context
- Health ≤ 5 → immediate DQN fallback for survival
- Health < 30% → re-query for health management

### Running Training
```powershell
# Prerequisites: Conda env + LM Studio on http://127.0.0.1:1234
conda activate HeRoN

# Full HeRoN (DQN + Helper + Reviewer)
cd training; python heron_training.py

# DQN-only baseline for comparison
python DQN_training.py

# Dataset generation (50 episodes → ~2500 samples)
cd "dataset Reviewer"; python crafter_dataset_generation.py

# Reviewer fine-tuning (5 epochs, saves to reviewer_retrained/)
cd reviewer_fine_tuning; python reviewer_fine_tuning.py

# Evaluation report (HeRoN vs baselines)
cd evaluation; python evaluation_report_generator.py
```

## Critical Patterns

### State Handling (MUST USE)
```python
# Environment extracts state from info dict - DO NOT use raw RGB observation
state = env.reset()  # Returns 43-dim vector, NOT image
next_state, reward, done, info = env.step(action)

# DQN requires PyTorch tensor on correct device
state_tensor = torch.FloatTensor(state).to(agent.device)  # Auto-detects CUDA/CPU
q_values = agent.model(state_tensor)
```

### Model Persistence (4 Files)
```python
# Saves: .pth (weights), _memory.pkl, _priorities.pkl, _epsilon.txt
agent.save("models/crafter_heron_final")  # Relative path, auto-creates dirs
agent.load("checkpoints/best_model_ep42_ach15")

# Performance-based checkpointing
if episode_achievements > best_count:
    agent.save(f"checkpoints/best_model_ep{e}_ach{achievements}")
```

### LLM Response Parsing
```python
# Expected format: "[move_right], [do], [move_left]" (3-5 actions)
llm_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)
action_ids = [CrafterHelper.ACTION_ID_MAP.get(m.strip().lower()) 
              for m in re.findall(r'\[(.*?)\]', llm_response)[:5]]

# Typo correction (13 mappings)
typo_map = {"place_rock": "place_stone", "make_pickaxe": "make_wood_pickaxe"}
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
# ❌ WRONG - Decays to 0 in first episode!
while not done:
    threshold -= 0.01  # Inside step loop

# ✅ CORRECT - Decay per episode
for episode in range(episodes):
    # ... episode loop
    if episode < threshold_episodes:
        threshold = max(0, threshold - 0.01)  # After episode ends
```

### 3. LM Studio Connectivity
```powershell
# Verify server is running and model loaded
(Invoke-WebRequest -Uri "http://127.0.0.1:1234/v1/models").Content | ConvertFrom-Json
# Returns {"data": [{"id": "llama-3.2-3b-instruct", ...}]}
```

### 4. State Dimensions
```python
# MUST match environment
assert env.state_size == 43 and env.action_size == 17
agent = DQNAgent(state_size=43, action_size=17)  # NOT 36/9 (deprecated Battle env)
```

### 5. Reviewer Fallback
```python
# Training continues gracefully if reviewer_retrained/ missing
# Run `reviewer_fine_tuning.py` first for full three-agent system
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
- `classes/crafter_environment.py`: 43-dim state extraction from Crafter
- `classes/agent.py`: Double DQN + prioritized replay (PyTorch)
- `classes/crafter_helper.py`: LLM sequence generator + re-planning logic
- `classes/instructor_agent.py`: T5 Reviewer
- `training/heron_training.py`: Three-agent training loop
- `training/DQN_training.py`: DQN-only baseline
- `evaluation/evaluation_system.py`: Metrics aggregation + convergence detection

## Development Notes

### Reward Shaping
Track `native_reward` (sparse +1/achievement) and `shaped_reward` separately for ablation studies:
```python
shaped_reward = native_reward + resource_bonus(+0.1) + health_bonus(+0.05) + tier_bonus(+0.05) + tool_bonus(+0.02)
```

### Dependencies
- PyTorch (CUDA/CPU, no MPS support)
- LM Studio (local LLM server on :1234)
- Crafter environment (`pip install crafter>=1.8.3`)
- Transformers (T5 Reviewer fine-tuning)

### Testing
```powershell
# Test environment wrapper
python test_crafter_env.py

# Test LM Studio connection
python test_lmstudio_connection.py
```