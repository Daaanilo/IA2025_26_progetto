# HeRoN - Multi-Agent RL-LLM Framework for Game AI

## Project Overview
HeRoN integrates three agents for adaptive NPC decision-making in survival crafting games:
- **DQNAgent**: Parametric neural network (state_size=41, action_size=17) learns from experience replay
- **CrafterHelper**: Zero-shot LLM generates 3-5 action sequences via LM Studio API
- **InstructorAgent**: Fine-tuned T5 refines Helper suggestions before execution

**Current Focus**: Crafter environment (22 achievements, sparse rewards). Battle environment deprecated.

## Core Architecture

### Crafter Environment (`classes/crafter_environment.py`)
State vector (41 dims): `[inventory(13), position(2), status(3), achievements(22), fence(1)]`
- Wraps `crafter.Env()` with semantic feature extraction from `info` dict
- All 17 actions always valid (no masking like Battle environment)
- Sparse rewards: +1 per achievement unlock, 0 otherwise
- **Critical**: Call `_extract_state()` to convert RGB observation → numerical vector

### Three-Agent Training Loop (`HeRoN/heron_crafter.py`)
```python
# Probability threshold decay: 1.0 → 0.0 over 10 episodes
if random() > threshold and episode < 600:
    # LLM workflow
    sequence = helper.generate_action_sequence(state, info)
    if reviewer_available:
        feedback = reviewer.generate_suggestion(state_desc, helper_response)
        refined_sequence = helper.generate_action_sequence(...)  # Re-prompt with feedback
    executor.current_sequence = refined_sequence
else:
    # Pure DQN after threshold decay
    action = agent.act(state, env)
```

**Key Pattern**: SequenceExecutor manages 3-5 action sequences with re-planning triggers:
- Achievement unlock → interrupt, re-query LLM
- Critical health (<20%) → DQN fallback for survival
- Resource depletion → interrupt, new sequence

### DQN Agent (`classes/agent.py`)
Architecture: `Dense(128) → Dense(128) → Dense(64) → Dense(action_size)`
- **State reshaping**: Always use `np.reshape(state, [1, env.state_size])` before `model.predict()`
- **Memory replay**: Only trigger when `len(agent.memory) > batch_size`
- **Model persistence**: Saves 3 files: `{prefix}.keras`, `{prefix}_memory.pkl`, `{prefix}_epsilon.txt`
- **Absolute paths**: All save/load uses `f"/{path_prefix}.keras"` (update before training)

## Development Workflows

### Running Training Experiments
```bash
conda activate HeRoN
cd HeRoN
python heron_crafter.py  # Main three-agent training
```

**Prerequisites**:
1. LM Studio running on `http://127.0.0.1:1234` with model loaded (default: `llama-3.2-3b-instruct`)
2. Update `REVIEWER_MODEL_PATH` in `heron_crafter.py` (placeholder until F06 completion)
3. Set output paths for CSV/PNG before running (search for `f"heron_crafter_"` prefix strings)

### Baseline Comparisons
```bash
cd evaluation
python baseline_crafter_dqn.py      # Pure DQN (no LLM)
python baseline_crafter_helper.py   # LLM-only (no DQN learning)
```
Use `evaluation_report_generator.py` to generate markdown comparison reports.

### Dataset Generation for Reviewer Fine-Tuning
```bash
cd "dataset Reviewer"
python crafter_dataset_generation.py  # Generates game_scenarios_dataset_crafter.jsonl
```
**Output**: JSONL (one JSON object per line) with fields `[episode_id, step, prompt, response, instructions, quality_score, achievements_unlocked]`
- Default: 50 episodes × ~50 Helper calls/episode = ~2500 samples
- Adjust `NUM_EPISODES` for larger datasets (100 episodes → 5000 samples)

### Windows-Specific: Installing Crafter
```powershell
$env:PYTHONUTF8=1
pip install crafter>=1.8.3
```
Crafter requires UTF-8 encoding; set environment variable before installation.

## Project-Specific Patterns

### LLM Response Parsing (Crafter)
```python
# Multi-action sequences: "[move_right], [do], [move_left]"
matches = re.findall(r'\[(.*?)\]', llm_response)  # Extracts all bracketed actions
action_sequence = [ACTION_ID_MAP.get(match.strip().lower()) for match in matches[:5]]

# Clean thinking tags before parsing
llm_response = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL)
```
**Typo handling** (`crafter_helper.py`): 13 common typo mappings like `"place_rock" → "place_stone"`

### Reward Shaping System (`heron_crafter.py`)
Crafter's native rewards are sparse (+1 per achievement only). HeRoN adds intrinsic bonuses:
```python
shaped_reward = native_reward + bonus_resource(+0.1) + bonus_health(+0.05) 
                + bonus_tier_progression(+0.05) + bonus_tool_usage(+0.02)
```
**Critical**: Track `native_reward` and `shaped_reward` separately for ablation studies.

### State Description for LLM Context
```python
# Convert 41-dim vector → human-readable prompt
inventory_str = ", ".join([f"{k}: {v}" for k, v in inventory.items() if v > 0])
current_goal = helper._determine_current_goal(inventory, achievements, unlocked)
description = f"Inventory: {inventory_str}\nNext Priority: {current_goal}"
```
Helper uses heuristic goal prioritization (collect wood → stone → iron → craft pickaxe).

### Re-planning Triggers (Strategy B)
SequenceExecutor interrupts 3-5 action sequences when:
1. **Achievement unlocked**: New achievement mid-sequence → re-query Helper
2. **Health critical**: `health < 0.2 * max_health` → DQN fallback for survival
3. **Resource depleted**: Critical resource (wood/stone/iron) drops to 0 → re-query

```python
if helper.should_replan(state, info, previous_info, action_sequence):
    executor.interrupt_sequence()  # Fallback to DQN for remaining actions
```

## Development Guidelines

### Critical State Management Patterns
```python
# ALWAYS reshape state before DQN prediction
state = np.reshape(state, [1, env.state_size])  # Shape: (1, 41)
action = agent.act(state, env)

# Memory replay only when sufficient samples
if len(agent.memory) > batch_size:
    agent.replay(batch_size, env)

# Threshold decay with floor at 0
threshold = max(0, threshold - 0.1)  # Prevents negative values
```

### Model Persistence (CRITICAL)
All paths use absolute format with leading `/`:
```python
agent.save("/path/to/crafter_heron_final")  # Creates 3 files: .keras, _memory.pkl, _epsilon.txt
agent.load("/path/to/crafter_heron_final")  # Load all 3 files

# BEFORE TRAINING: Update these absolute paths in your script
# Search for: f"/{path_prefix}" or f"heron_crafter_"
```

### Plotting and CSV Export Conventions
Every experiment generates 5+ PNG plots and 1 CSV file:
```python
# Standard outputs (update paths before running)
heron_crafter_rewards.png          # Native + shaped + bonus trends
heron_crafter_achievements.png     # Cumulative unlocks
heron_crafter_moves.png            # Episode length
heron_crafter_helper_stats.png     # Helper calls + hallucination rate
heron_crafter_metrics.csv          # All per-episode data
```

### Device Selection (Cross-Platform)
```python
device = torch.device("mps" if torch.backends.mps.is_available()      # Apple M-series
                      else "cuda" if torch.cuda.is_available()        # NVIDIA GPU
                      else "cpu")                                     # Fallback
```

## Common Pitfalls

### 1. Model Path Errors (Most Common)
```python
# ❌ WRONG - Will fail silently or create files in system root
agent.save("model")

# ✅ CORRECT - Absolute path with leading /
agent.save("/d/Progetto_AI2/HeRoN/models/crafter_heron_final")
```
**Before ANY training run**: Search codebase for `f"/{path_prefix}"` and `f"heron_crafter_"` to update paths.

### 2. LM Studio Connection Failures
```bash
# Error: Connection refused on port 1234
# Solution: Start LM Studio server BEFORE running training
# Verify: http://127.0.0.1:1234/v1/models should return JSON
```

### 3. Reviewer Model Placeholder
Current `heron_crafter.py` has placeholder paths:
```python
REVIEWER_MODEL_PATH = "path/to/flan-t5-crafter-fine-tuned"  # TODO: Update after F06
```
Training gracefully continues without Reviewer if model unavailable, but log warnings.

### 4. State Size Mismatches
```python
# Crafter: state_size=41, action_size=17
# Battle (deprecated): state_size=36, action_size=9
# Ensure DQNAgent initialized with correct sizes
agent = DQNAgent(state_size=41, action_size=17)  # For Crafter
```

### 5. Episode Reset Without Cleanup
```python
# ❌ MISSING - Causes incorrect tracking
state = env.reset()
# ... training loop

# ✅ CORRECT - Reset episode statistics
state = env.reset()
episode_reward = 0
episode_achievements = set()
previous_info = None
```

## Feature Development Workflow

### Feature Implementation Process
1. Check `features.md` for feature specifications and implementation notes
2. When implementing features, mark them as completed (✓) in `features.md` and wrote the relative implementation notes
3. **Always update `modifiche.md`** after feature completion:
   - Write feature ID (e.g., "F04: Prompt Engineering Helper")
   - Describe implementation approach and key decisions
   - Include file diff summary and trade-offs documented
   - Note any compatibility or architectural changes

### Current Feature Status
- ✅ **F01-F05**: Crafter environment integration complete
- ✅ **F08**: Three-agent training loop complete (with F06 placeholder)
- ✅ **F10**: Evaluation system complete
- ⏳ **F06**: Reviewer fine-tuning (in progress - model paths are placeholders)
- ⏳ **F07**: Optimal sequence length analysis
- ⏳ **F09-F14**: Iterative training, testing, documentation

### Testing Strategy Pattern
Create dedicated test files for new components:
```bash
test_crafter_env.py         # Environment wrapper validation
test_f04_helper.py          # Helper sequence generation
test_lmstudio_connection.py # LLM API connectivity
```
Run tests before committing features to verify integration.