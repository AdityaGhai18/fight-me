import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from src.env_iterations.custom_env import StreetFighter

LOG_DIR = '../../data/logs/logs/'
os.makedirs(LOG_DIR, exist_ok=True)

# Create the environment WITHOUT render_mode="human"
env = StreetFighter()
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)

# Initialize the model with improved hyperparameters
model = PPO(
    policy='CnnPolicy',
    env=env,
    learning_rate=1e-4,  # Slightly lower for stability
    n_steps=2048,        # Keep large for better exploration
    batch_size=64,       # Can increase if you have more RAM/GPU
    n_epochs=10,         # Default is fine, but 10-15 is common
    gamma=0.99,          # Standard for episodic tasks
    gae_lambda=0.95,     # Standard
    clip_range=0.2,      # Standard
    ent_coef=0.001,      # Lower than before to reduce randomness
    max_grad_norm=0.5,   # Helps prevent exploding gradients
    vf_coef=0.5,         # Default, but can be tuned
    verbose=1
)

# Train the model for a realistic number of steps
total_timesteps = 1_000_000  # 10,000 is way too low for complex games[1][2][3]
model.learn(total_timesteps=total_timesteps, progress_bar=True)

# Save the model
model.save("../../data/models/ppo_streetfighter")

# Optional: Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Close the environment (with error handling for macOS/pyglet)
try:
    env.close()
except Exception as e:
    print("Error closing environment:", e)
