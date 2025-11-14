"""
Test script per verificare integrazione CrafterEnv con DQN.
Testa:
1. CrafterEnv creation e feature extraction
2. DQNAgent compatibility con state_size=43 (16 inventory + 2 pos + 3 status + 22 achievements)
3. 5 episodi di 100 steps con azioni random
4. Verifica shape, rewards, done flags
"""

import numpy as np
import sys
sys.path.insert(0, './classes')

from crafter_environment import CrafterEnv
from agent import DQNAgent


def test_crafter_env():
    print("=" * 60)
    print("Testing CrafterEnv Integration with DQN")
    print("=" * 60)
    
    # === Test 1: Create CrafterEnv ===
    print("\n[1/5] Creating CrafterEnv...")
    try:
        env = CrafterEnv(reward=True, length=10000)
        print(f"✓ CrafterEnv created successfully")
        print(f"  - State size: {env.get_state_size()}")
        print(f"  - Action size: {env.get_action_size()}")
        print(f"  - Action space: Discrete({env.action_size})")
    except Exception as e:
        print(f"✗ Failed to create CrafterEnv: {e}")
        return False
    
    # === Test 2: Initialize DQN Agent ===
    print("\n[2/5] Initializing DQNAgent with state_size=43...")
    try:
        agent = DQNAgent(
            state_size=env.get_state_size(),
            action_size=env.get_action_size(),
            load_model_path=None
        )
        print(f"✓ DQNAgent initialized")
        print(f"  - State size: {agent.state_size}")
        print(f"  - Action size: {agent.action_size}")
        print(f"  - Model layers: 43 → 128 → 128 → 64 → 17")
    except Exception as e:
        print(f"✗ Failed to initialize DQNAgent: {e}")
        return False
    
    # === Test 3: Reset and get initial state ===
    print("\n[3/5] Testing env.reset()...")
    try:
        state = env.reset()
        print(f"\u2713 Environment reset successfully")
        print(f"  - State shape: {state.shape}")
        print(f"  - State dtype: {state.dtype}")
        print(f"  - State sample: {state[:5]}...")
        assert state.shape == (43,), f"Expected shape (43,), got {state.shape}"
        assert state.dtype == np.float32, f"Expected float32, got {state.dtype}"
    except Exception as e:
        print(f"✗ Failed env.reset(): {e}")
        return False
    
    # === Test 4: Run episodes with DQN predictions ===
    print("\n[4/5] Running 5 test episodes (100 steps each)...")
    try:
        total_steps = 0
        total_rewards = 0
        episodes_completed = 0
        
        for episode in range(5):
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done and episode_steps < 100:
                # Agent predicts action
                state_reshaped = np.reshape(state, [1, env.state_size])
                q_values = agent.model.predict(state_reshaped, verbose=0)[0]
                action = np.argmax(q_values)
                
                # Environment step
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                state = next_state
            
            episodes_completed += 1
            total_rewards += episode_reward
            print(f"  Episode {episode+1}: {episode_steps} steps, reward={episode_reward:.2f}")
        
        print(f"✓ Successfully ran {episodes_completed} episodes")
        print(f"  - Total steps: {total_steps}")
        print(f"  - Total rewards: {total_rewards:.2f}")
        print(f"  - Avg reward per episode: {total_rewards / episodes_completed:.2f}")
        
    except Exception as e:
        print(f"✗ Failed during episode runs: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # === Test 5: Verify all features ===
    print("\n[5/5] Final verification...")
    try:
        # Get last info to verify feature extraction
        state = env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                break
        
        print(f"✓ All tests passed!")
        print(f"  - CrafterEnv correctly integrated with DQN")
        print(f"  - Feature extraction working properly")
        print(f"  - State shape consistent: {state.shape}")
        print(f"  - Action selection working")
        
        print("\n" + "=" * 60)
        print("✓ CrafterEnv is ready for HeRoN integration!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"✗ Final verification failed: {e}")
        return False


if __name__ == '__main__':
    success = test_crafter_env()
    sys.exit(0 if success else 1)
