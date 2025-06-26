import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from src.env_iterations.custom_env3 import StreetFighter
import logging
from datetime import datetime
import json
import multiprocessing
from stable_baselines3.common.vec_env import VecEnv
import signal
import sys
import psutil

# Set up logging
def setup_logging():
    os.makedirs('../../data/logs/logs3', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'../../data/logs/logs3/training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def make_env(rank):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        try:
            env = StreetFighter()
            # Set a different seed for each environment to ensure variety
            env.reset(seed=rank + 42)  # Use rank + 42 to ensure different seeds
            env = Monitor(env)
            return env
        except Exception as e:
            logging.error(f"Error creating environment {rank}: {str(e)}")
            raise
    return _init

class BestModelSavingCallback(BaseCallback):
    def __init__(self, check_freq=10000, save_freq=50000, verbose=1):
        super(BestModelSavingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_freq = save_freq
        self.best_mean_reward = -np.inf
        self.best_model_path = None
        self.last_save_time = datetime.now()

    def _on_step(self):
        try:
            # Regular checkpoint saving
            if self.n_calls % self.save_freq == 0:
                checkpoint_path = f"../../data/models/checkpoints3/model_{self.n_calls}"
                self.model.save(checkpoint_path)
                logging.info(f"Saved checkpoint at step {self.n_calls}")
                self.last_save_time = datetime.now()

            # Best model evaluation and saving
            if self.n_calls % self.check_freq == 0:
                # Log system resources
                process = psutil.Process()
                memory_info = process.memory_info()
                logging.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
                
                # Evaluate current model with more episodes and some randomness
                mean_reward, std_reward = evaluate_policy(
                    self.model, 
                    self.training_env, 
                    n_eval_episodes=20,  # Increased from 10 to 20
                    deterministic=False  # Use stochastic actions for realistic evaluation
                )
                
                # Additional detailed evaluation
                episode_rewards = []
                total_episodes = 0
                
                # Run a few more episodes to get detailed stats
                for ep in range(20):
                    obs = self.training_env.reset()
                    done = False
                    total_reward = 0
                    info = {}
                    
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=False)  # Use stochastic actions
                        obs, reward, done, info = self.training_env.step(action)
                        # Handle vectorized environment
                        if isinstance(reward, (list, np.ndarray)):
                            reward = reward[0]
                        if isinstance(done, (list, np.ndarray)):
                            done = done[0]
                        if isinstance(info, (list, tuple)):
                            info = info[0]
                        total_reward += reward
                    
                    episode_rewards.append(total_reward)
                    total_episodes += 1
                
                # Log evaluation results
                logging.info(f"Evaluation at step {self.n_calls}:")
                logging.info(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                logging.info(f"  Detailed stats (20 episodes):")
                logging.info(f"    Avg episode reward: {np.mean(episode_rewards):.1f}")
                logging.info(f"    Min/Max reward: {np.min(episode_rewards):.1f}/{np.max(episode_rewards):.1f}")
                
                # Use normalized episode rewards for determining best model (not raw VecNormalize values)
                normalized_avg_reward = np.mean(episode_rewards)
                
                # Save if this is the best model so far (using normalized rewards)
                if normalized_avg_reward > self.best_mean_reward:
                    self.best_mean_reward = normalized_avg_reward
                    best_model_path = f"../../data/models/best_model3/best_model_{self.n_calls}"
                    self.model.save(best_model_path)
                    self.best_model_path = best_model_path
                    logging.info(f"New best model saved! Normalized avg reward: {normalized_avg_reward:.2f}")
                    logging.info(f"Previous best: {self.best_mean_reward:.2f}")
                    
                    # Also save a copy with a fixed name for easy access
                    self.model.save("../../data/models/best_model3/latest_best_model")
        except Exception as e:
            logging.error(f"Error in callback: {str(e)}")
            logging.error(f"Error type: {type(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
        return True

def cleanup_resources():
    """Clean up any remaining processes"""
    try:
        # Kill any remaining Python processes
        for proc in psutil.process_iter(['pid', 'name']):
            if 'python' in proc.info['name'].lower():
                try:
                    proc.kill()
                except:
                    pass
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    logging.info("Received interrupt signal, cleaning up...")
    cleanup_resources()
    sys.exit(0)

def main():
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Setup logging
    log_file = setup_logging()
    logging.info("Starting training with optimized parameters")
    logging.info(f"Log file: {log_file}")

    # Create directories for saving models
    os.makedirs('../../data/models/best_model3', exist_ok=True)
    os.makedirs('../../data/models/checkpoints3', exist_ok=True)

    # Number of parallel environments
    n_cpu = multiprocessing.cpu_count()
    n_envs = min(4, n_cpu)  # Use 4 environments or less
    logging.info(f"Using {n_envs} parallel environments")

    env = None
    model = None
    best_model_callback = None
    try:
        # Create parallel environments
        logging.info("Creating parallel environments...")
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        env = VecFrameStack(env, n_stack=4)
        env = VecNormalize(env, norm_obs=False, norm_reward=True)
        logging.info("Parallel environments created successfully")

        # Update these parameters to match Trial 13 from Optuna optimization
        ppo_params = {
            'learning_rate': 0.000133431518150732,
            'n_steps': 1274,
            'batch_size': 380,
            'n_epochs': 8,
            'gamma': 0.99,
            'gae_lambda': 0.9698577716505229,
            'clip_range': 0.22623031805757216,
            'ent_coef': 0.010691267335158192,
            'max_grad_norm': 0.8072142963373791,
            'vf_coef': 0.6632027254254521,
            'policy_kwargs': dict(
                net_arch=dict(
                    pi=[152, 166],
                    vf=[205, 138]
                ),
                normalize_images=False  # Don't normalize images since VecNormalize does it
            )
        }

        # Optimized parameters from Trial 13
        logging.info("Initializing model with Trial 13 optimized parameters...")
        model = PPO('CnnPolicy', env, **ppo_params)
        logging.info("Model initialized successfully")

        # Set up callback for saving best model
        logging.info("Setting up best model saving callback...")
        best_model_callback = BestModelSavingCallback(
            check_freq=10000,
            save_freq=50000
        )

        # Train the model
        logging.info("Starting training...")
        TIMESTEPS = 10_000_000
        model.learn(
            total_timesteps=TIMESTEPS,
            callback=best_model_callback,
            progress_bar=True
        )

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise
    finally:
        # Ensure environments are properly closed
        if env is not None:
            logging.info("Closing training environment...")
            try:
                env.close()
            except Exception as e:
                logging.error(f"Error closing environment: {str(e)}")
        
        # Clean up any remaining resources
        cleanup_resources()

    # Create evaluation environment (single environment for evaluation)
    logging.info("Creating evaluation environment...")
    eval_env = StreetFighter()
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True)
    
    # Sync the evaluation environment's normalization with the training environment
    # Only sync if training completed successfully and env exists
    if env is not None and hasattr(env, 'obs_rms') and hasattr(eval_env, 'obs_rms'):
        eval_env.obs_rms = env.obs_rms
    if env is not None and hasattr(env, 'ret_rms') and hasattr(eval_env, 'ret_rms'):
        eval_env.ret_rms = env.ret_rms

    try:
        # Final evaluation
        logging.info("\nPerforming final evaluation...")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=False)

        # Save the final model
        logging.info("Saving final model...")
        model.save("ppo_streetfighter_final3")

        # Log the best model information
        if best_model_callback and best_model_callback.best_model_path:
            logging.info(f"\nBest model was saved at: {best_model_callback.best_model_path}")
            logging.info(f"Best mean reward achieved: {best_model_callback.best_mean_reward:.2f}")
            logging.info("You can also find the latest best model at: ../../data/models/best_model3/latest_best_model")

    finally:
        # Ensure evaluation environment is closed
        eval_env.close()
        logging.info("Training completed successfully")

if __name__ == "__main__":
    main() 