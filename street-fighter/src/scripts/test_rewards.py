import numpy as np
from src.env_iterations.custom_env3 import StreetFighter
import matplotlib.pyplot as plt
from collections import defaultdict
import time

def test_rewards(n_episodes=10):
    env = StreetFighter()
    rewards_log = defaultdict(list)
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_rewards = []
        health_rewards = []
        score_rewards = []
        loss_penalties = []
        
        # Track previous values
        prev_health = info.get('health', 176)
        prev_score = info.get('score', 0)
        prev_enemy_matches = info.get('enemy_matches_won', 0)
        
        while not done:
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Calculate rewards before any resets
            current_health = info.get('health', 176)
            current_score = info.get('score', 0)
            current_enemy_matches = info.get('enemy_matches_won', 0)
            
            # Calculate components
            health_reward = (current_health - prev_health) * 10
            score_reward = current_score - prev_score
            loss_penalty = -5000 if current_enemy_matches > prev_enemy_matches else 0
            
            # Update previous values
            prev_health = current_health
            prev_score = current_score
            prev_enemy_matches = current_enemy_matches
            
            # Log rewards
            health_rewards.append(health_reward)
            score_rewards.append(score_reward)
            loss_penalties.append(loss_penalty)
            episode_rewards.append(reward)
            time.sleep(0.00005)
        
        # Log episode statistics
        rewards_log['total_reward'].append(sum(episode_rewards))
        rewards_log['health_reward'].append(sum(health_rewards))
        rewards_log['score_reward'].append(sum(score_rewards))
        rewards_log['loss_penalty'].append(sum(loss_penalties))
        
        print(f"\nEpisode {episode + 1}:")
        print(f"Total: {sum(episode_rewards):.2f}")
        print(f"Health: {sum(health_rewards):.2f}")
        print(f"Score: {sum(score_rewards):.2f}")
        print(f"Loss: {sum(loss_penalties):.2f}")
        print(f"Final Health: {current_health}")
        print(f"Final Time: {info.get('time', 0)}")
        
        # Print max/min rewards to see the range
        print(f"Max Health Reward: {max(health_rewards):.2f}")
        print(f"Min Health Reward: {min(health_rewards):.2f}")
        print(f"Max Score Reward: {max(score_rewards):.2f}")
        print(f"Min Score Reward: {min(score_rewards):.2f}")
    
    # Print summary statistics
    print("\nOverall Statistics:")
    for key, values in rewards_log.items():
        print(f"{key}:")
        print(f"  Mean: {np.mean(values):.2f}")
        print(f"  Std: {np.std(values):.2f}")
        print(f"  Min: {np.min(values):.2f}")
        print(f"  Max: {np.max(values):.2f}")

if __name__ == "__main__":
    test_rewards() 