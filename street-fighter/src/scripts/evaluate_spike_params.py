import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from src.env_iterations.custom_env2 import StreetFighter
import json


params = {
    'learning_rate': 0.00044990344843925825,
    'n_steps': 7465,
    'batch_size': 96,
    'n_epochs': 24,
    'gamma': 0.982159118724112,
    'gae_lambda': 0.9178695193792474,
    'clip_range': 0.267444141967346,
    'ent_coef': 0.00011481472935802298,
    'max_grad_norm': 0.3637677054024369,
    'vf_coef': 0.4741413121652922,
    'net_arch_pi': [122, 383, 147],
    'net_arch_vf': [90, 330, 187],
}


N_EVAL_EPISODES = 10
TIMESTEPS = 250000  # 250k timesteps


def main():
    # Create the environment
    env = StreetFighter()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    # Create the PPO model with the given parameters
    model = PPO(
        policy='CnnPolicy',
        env=env,
        learning_rate=params['learning_rate'],
        n_steps=params['n_steps'],
        batch_size=params['batch_size'],
        n_epochs=params['n_epochs'],
        gamma=params['gamma'],
        gae_lambda=params['gae_lambda'],
        clip_range=params['clip_range'],
        ent_coef=params['ent_coef'],
        max_grad_norm=params['max_grad_norm'],
        vf_coef=params['vf_coef'],
        policy_kwargs=dict(
            net_arch=dict(
                pi=params['net_arch_pi'],
                vf=params['net_arch_vf']
            )
        ),
        verbose=1
    )

    print(f"Training PPO with spike parameters for {TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True)

    print(f"\nEvaluating over {N_EVAL_EPISODES} episodes...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Std Reward:  {std_reward:.2f}")

    # Save results
    results = {
        'params': params,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'n_eval_episodes': N_EVAL_EPISODES,
        'timesteps': TIMESTEPS
    }
    with open('../../data/logs/spike_eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to ../../data/logs/spike_eval_results.json")

if __name__ == "__main__":
    main() 