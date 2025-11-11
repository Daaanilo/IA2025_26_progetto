"""
Step 3: Collect Reviewer Training Data

Obiettivo: Creare un dataset specifico per l'ambiente Crafter che contenga:
- Stati dell'ambiente
- Azioni suggerite dall'Helper
- Feedback correttivi e mirati

Questo script raccoglie interazioni tra Helper e un Reviewer base (o mock)
per generare dati di training per il fine-tuning del Reviewer.

Usage (PowerShell):
  conda activate ia2025
  python scripts/3_collect_reviewer_data.py --episodes 200 --output data/reviewer_dataset.jsonl
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import time

import yaml

# Add repo root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.environment import make_crafter_env
from src.llm import Helper, Reviewer


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser("Collect Helper-Reviewer interactions for fine-tuning")
    p.add_argument('--episodes', type=int, default=200, 
                   help='Number of episodes to collect data from')
    p.add_argument('--steps-per-episode', type=int, default=1000, 
                   help='Max steps per episode')
    p.add_argument('--output', type=str, default='data/reviewer_dataset.jsonl', 
                   help='Output JSONL file (will be appended)')
    p.add_argument('--helper-config', type=str, default='configs/helper_config.yaml',
                   help='Helper configuration file')
    p.add_argument('--reviewer-config', type=str, default='configs/reviewer_config.yaml',
                   help='Reviewer configuration file')
    p.add_argument('--dqn-config', type=str, default='configs/dqn_config.yaml',
                   help='DQN configuration file')
    p.add_argument('--helper-query-freq', type=int, default=100, 
                   help='Query Helper every N steps')
    p.add_argument('--use-baseline-npc', type=str, default=None,
                   help='Path to baseline NPC model for generating realistic trajectories')
    return p.parse_args()


def ensure_dir_for_file(path: str):
    """Create directory for file if it doesn't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def generate_expert_feedback(state_info: dict, suggested_actions: list, 
                            actual_outcome: dict = None) -> dict:
    """
    Generate expert feedback for suggested actions.
    
    This creates synthetic feedback based on game state and suggested actions.
    In a real scenario, this could be:
    - Human expert annotations
    - Outcome-based feedback
    - Rule-based expert system
    
    Args:
        state_info: Current state information
        suggested_actions: Actions suggested by Helper
        actual_outcome: Outcome after executing actions (if available)
        
    Returns:
        Dictionary with rating and feedback
    """
    feedback = {
        'rating': 5,  # Default neutral rating (1-10 scale)
        'feedback': '',
        'improved_actions': suggested_actions.copy(),
        'reasoning': ''
    }
    
    # Extract state info
    health = state_info.get('health', 9)
    food = state_info.get('food', 9)
    drink = state_info.get('drink', 9)
    inventory = state_info.get('inventory', {})
    
    # Analyze suggestions and provide feedback
    action_analysis = {
        'has_movement': any('move' in a for a in suggested_actions),
        'has_crafting': any('make' in a for a in suggested_actions),
        'has_placement': any('place' in a for a in suggested_actions),
        'has_gathering': 'do' in suggested_actions,
    }
    
    reasoning_parts = []
    
    # Rule 1: Low health/food/drink should prioritize survival
    if health < 5 or food < 5 or drink < 5:
        if not action_analysis['has_gathering'] and not 'sleep' in suggested_actions:
            feedback['rating'] = 4
            reasoning_parts.append("Low survival stats detected, should prioritize gathering resources or sleeping")
            # Suggest improvements
            if 'noop' in suggested_actions:
                idx = suggested_actions.index('noop')
                feedback['improved_actions'][idx] = 'do' if food < 5 else 'sleep'
        else:
            feedback['rating'] = 7
            reasoning_parts.append("Good prioritization of survival needs")
    
    # Rule 2: Crafting without resources is problematic
    if action_analysis['has_crafting']:
        crafting_actions = [a for a in suggested_actions if 'make' in a]
        for craft_action in crafting_actions:
            if 'wood' in craft_action and inventory.get('wood', 0) == 0:
                feedback['rating'] = max(feedback['rating'] - 2, 1)
                reasoning_parts.append(f"Suggesting {craft_action} without wood in inventory")
            elif 'stone' in craft_action and inventory.get('stone', 0) == 0:
                feedback['rating'] = max(feedback['rating'] - 2, 1)
                reasoning_parts.append(f"Suggesting {craft_action} without stone in inventory")
            elif 'iron' in craft_action and inventory.get('iron', 0) == 0:
                feedback['rating'] = max(feedback['rating'] - 2, 1)
                reasoning_parts.append(f"Suggesting {craft_action} without iron in inventory")
    
    # Rule 3: Good progression - gathering before crafting
    if action_analysis['has_gathering'] and action_analysis['has_crafting']:
        gather_idx = [i for i, a in enumerate(suggested_actions) if a == 'do']
        craft_idx = [i for i, a in enumerate(suggested_actions) if 'make' in a]
        if gather_idx and craft_idx and min(gather_idx) < min(craft_idx):
            feedback['rating'] = min(feedback['rating'] + 2, 10)
            reasoning_parts.append("Good action sequence: gathering before crafting")
    
    # Rule 4: Too many redundant actions
    unique_actions = len(set(suggested_actions))
    if unique_actions < len(suggested_actions) * 0.5:
        feedback['rating'] = max(feedback['rating'] - 1, 1)
        reasoning_parts.append("Too many redundant actions in sequence")
    
    # Rule 5: Movement without purpose
    movement_count = sum(1 for a in suggested_actions if 'move' in a)
    if movement_count > len(suggested_actions) * 0.6:
        feedback['rating'] = max(feedback['rating'] - 1, 1)
        reasoning_parts.append("Excessive movement without clear purpose")
    elif movement_count > 0:
        feedback['rating'] = min(feedback['rating'] + 1, 10)
        reasoning_parts.append("Includes exploration movement")
    
    # Compile feedback
    feedback['reasoning'] = '; '.join(reasoning_parts) if reasoning_parts else "Reasonable action sequence"
    feedback['feedback'] = f"Rating {feedback['rating']}/10: {feedback['reasoning']}"
    
    return feedback


def main():
    """Main data collection loop."""
    args = parse_args()

    # Load configs
    print("Loading configurations...")
    with open(args.helper_config, 'r') as f:
        helper_cfg = yaml.safe_load(f)
    with open(args.reviewer_config, 'r') as f:
        reviewer_cfg = yaml.safe_load(f)
    with open(args.dqn_config, 'r') as f:
        dqn_cfg = yaml.safe_load(f)

    # Create environment
    print("\n" + "="*60)
    print("STEP 3: COLLECT REVIEWER TRAINING DATA")
    print("="*60)
    print("\nCreating Crafter environment...")
    env = make_crafter_env(dqn_cfg.get('environment'))

    # Initialize Helper
    print("Initializing Helper LLM...")
    helper = Helper(helper_cfg)
    
    # Initialize Reviewer (optional, for validation)
    print("Initializing Reviewer LLM...")
    try:
        reviewer = Reviewer(reviewer_cfg, device=reviewer_cfg.get('device', 'cuda'))
        reviewer.load_model(use_fine_tuned=False)
        print("Reviewer loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load Reviewer: {e}")
        print("Using rule-based feedback generation instead")
        reviewer = None

    # Load baseline NPC if provided
    npc_agent = None
    if args.use_baseline_npc:
        print(f"\nLoading baseline NPC from: {args.use_baseline_npc}")
        try:
            from src.agents import DQNAgent
            obs, _ = env.reset()
            npc_agent = DQNAgent(
                observation_shape=obs.shape,
                num_actions=env.action_space.n,
                config=dqn_cfg
            )
            npc_agent.load(args.use_baseline_npc)
            print("Baseline NPC loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load baseline NPC: {e}")
            npc_agent = None

    # Prepare output file
    out_path = args.output
    ensure_dir_for_file(out_path)

    print(f"\nCollecting data from {args.episodes} episodes")
    print(f"Output file: {out_path}")
    print(f"Helper query frequency: every {args.helper_query_freq} steps")
    
    total_samples = 0

    for ep in range(args.episodes):
        obs, _ = env.reset()
        step = 0
        action_history = []
        episode_samples = 0
        
        while step < args.steps_per_episode:
            # Query helper at intervals
            if step % args.helper_query_freq == 0:
                state_info = env.get_state_description()
                state_info['step'] = step
                state_info['episode'] = ep

                # Get recent actions for context
                recent_actions = [env.action_id_to_name(a) for a in action_history[-10:]] if action_history else []
                
                # Get Helper suggestions
                try:
                    suggested = helper.suggest_actions(
                        state_info=state_info, 
                        recent_actions=recent_actions
                    )
                    
                    # Generate feedback
                    if reviewer:
                        # Use actual Reviewer if available
                        feedback = reviewer.review_actions(
                            state_info=state_info, 
                            suggested_actions=suggested
                        )
                    else:
                        # Use rule-based feedback generation
                        feedback = generate_expert_feedback(
                            state_info=state_info,
                            suggested_actions=suggested
                        )
                    
                    # Create record
                    record = {
                        'episode': ep,
                        'step': step,
                        'timestamp': datetime.utcnow().isoformat(),
                        'state': state_info,
                        'suggested_actions': suggested,
                        'review_feedback': feedback,
                        'recent_actions': recent_actions
                    }
                    
                    # Append to file
                    with open(out_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    
                    episode_samples += 1
                    total_samples += 1
                    
                except Exception as e:
                    print(f"  Error collecting sample at episode {ep}, step {step}: {e}")
            
            # Take a step in environment
            if npc_agent:
                # Use baseline NPC action
                action = npc_agent.select_action(obs, epsilon=0.1)
            else:
                # Random action
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            action_history.append(action)
            done = terminated or truncated
            
            step += 1
            
            if done:
                break
        
        # Episode complete
        achievements = info.get('achievements', {})
        num_achievements = sum(1 for v in achievements.values() if v > 0)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{args.episodes} | "
                  f"Steps: {step} | "
                  f"Samples: {episode_samples} | "
                  f"Total Samples: {total_samples} | "
                  f"Achievements: {num_achievements}/22")
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETED")
    print("="*60)
    print(f"\nTotal samples collected: {total_samples}")
    print(f"Average samples per episode: {total_samples/args.episodes:.2f}")
    print(f"Dataset saved to: {out_path}")
    
    print("\nNext step: Run script 4 to fine-tune the Reviewer")


if __name__ == '__main__':
    main()
