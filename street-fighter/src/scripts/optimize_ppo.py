import os
import optuna
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from src.env_iterations.custom_env3 import StreetFighter
import logging
import json
from datetime import datetime
import time
import torch

N_TRIALS = 20
N_JOBS = 1

# Set up logging
def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('../../data/logs/optuna_logs', exist_ok=True)
    
    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'../../data/logs/optuna_logs/optimization_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    return log_file

def objective(trial):
    start_time = time.time()
    env = None
    
    try:
        # Log the start of a new trial
        logging.info(f"\n{'='*80}")
        logging.info(f"Starting Trial {trial.number}")
        logging.info(f"{'='*80}")
        
        # Create the environment
        logging.info("Creating environment...")
        env = StreetFighter()
        # Set a random seed for this trial to ensure different starting conditions
        env.reset(seed=trial.number)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=4)
        env = VecNormalize(env, norm_obs=False, norm_reward=True)
        logging.info("Environment created successfully")

        # Define hyperparameters to optimize with wider ranges and more options
        logging.info("Generating hyperparameters for this trial...")
        params = {
            # Learning parameters
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 3e-4, log=True),
            'n_steps': trial.suggest_int('n_steps', 1024, 4096),  # Increased range
            'batch_size': trial.suggest_int('batch_size', 128, 512),  # Increased range
            'n_epochs': trial.suggest_int('n_epochs', 5, 15),  # Increased range
            
            # PPO specific parameters
            'gamma': trial.suggest_float('gamma', 0.98, 0.999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.92, 0.98),
            'clip_range': trial.suggest_float('clip_range', 0.15, 0.25),
            'ent_coef': trial.suggest_float('ent_coef', 0.001, 0.05, log=True),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.5, 1),
            'vf_coef': trial.suggest_float('vf_coef', 0.4, 0.7),
        }
        
        n_layers = trial.suggest_int('n_layers', 2, 3)
        net_arch_pi = []
        net_arch_vf = []
        for i in range(n_layers):
            net_arch_pi.append(trial.suggest_int(f'net_arch_pi_{i}', 128, 512))
            net_arch_vf.append(trial.suggest_int(f'net_arch_vf_{i}', 128, 512))

        # Log the parameters for this trial
        logging.info("\nTrial Parameters:")
        logging.info("-" * 40)
        for key, value in params.items():
            logging.info(f"{key:15}: {value}")
        logging.info(f"policy_kwargs: {{'net_arch': {{'pi': {net_arch_pi}, 'vf': {net_arch_vf}}}}}")
        logging.info("-" * 40)

        # Create the model
        logging.info("\nCreating PPO model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                    pi=net_arch_pi,
                    vf=net_arch_vf
                ),
                normalize_images=False  # Don't normalize images since VecNormalize does it
            ),
            verbose=1,
            device=device
        )
        logging.info("Model created successfully")

        # Train the model with more timesteps
        logging.info("\nStarting training...")
        training_timesteps = 400_000  # Increased from 250,000
        logging.info(f"Training for {training_timesteps} timesteps")
        model.learn(total_timesteps=training_timesteps, progress_bar=True)
        training_time = time.time() - start_time
        logging.info(f"Training completed in {training_time:.2f} seconds")

        # Evaluate the model with more episodes
        logging.info("\nStarting evaluation...")
        logging.info("Running 20 evaluation episodes...")
        episode_rewards = []
        for ep in range(20):
            # Remove seed argument for vectorized envs
            obs = env.reset()
            done = False
            total_reward = 0
            info = {}
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, info = env.step(action)
                # SB3 vectorized envs return arrays, so extract scalar values
                if isinstance(reward, (np.ndarray, list)):
                    reward = reward[0]
                if isinstance(done, (np.ndarray, list)):
                    done = done[0]
                if isinstance(info, (list, tuple)):
                    info = info[0]
                total_reward += reward
            episode_rewards.append(total_reward)
            logging.info(f"Eval Ep {ep+1:2d}: Reward={total_reward:.2f}")
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        logging.info(f"Evaluation Results:")
        logging.info(f"  Mean Reward: {mean_reward:.2f}")
        logging.info(f"  Std Reward:  {std_reward:.2f}")
        logging.info(f"  Min/Max Reward: {np.min(episode_rewards):.2f}/{np.max(episode_rewards):.2f}")
        
        # Log trial summary
        total_time = time.time() - start_time
        logging.info(f"\nTrial {trial.number} Summary:")
        logging.info(f"  Total Time: {total_time:.2f} seconds")
        logging.info(f"  Training Time: {training_time:.2f} seconds")
        logging.info(f"  Final Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        logging.info(f"{'='*80}\n")

        return mean_reward

    except Exception as e:
        logging.error(f"Trial {trial.number} failed with error: {str(e)}")
        return float('-inf')  # Return negative infinity for failed trials
        
    finally:
        # Clean up environment
        if env is not None:
            env.close()

def main():
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting hyperparameter optimization")
    logging.info(f"Log file: {log_file}")
    
    # Create study with more thorough optimization
    logging.info("Creating Optuna study...")
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=5,  
            n_ei_candidates=16,   
            multivariate=True     
        )
    )
    
    logging.info(f"Beginning optimization with {N_TRIALS} trials and {N_JOBS} parallel jobs")
    logging.info("Each trial will:")
    logging.info("  1. Train for 400,000 timesteps")
    logging.info("  2. Evaluate over 20 episodes")
    logging.info("  3. Save results and parameters")
    logging.info(f"Estimated total time: {N_TRIALS * 20 / 60 / N_JOBS:.1f} hours (parallelized)")
    
    # n_jobs controls parallelization; set to 1 for serial, >1 for parallel
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)
    
    # Log the best parameters
    logging.info("\nOptimization Complete!")
    logging.info("=" * 40)
    logging.info("Best Trial Results:")
    logging.info(f"  Trial Number: {study.best_trial.number}")
    logging.info(f"  Best Value: {study.best_trial.value:.2f}")
    logging.info("\nBest Parameters:")
    for key, value in study.best_trial.params.items():
        logging.info(f"  {key:15}: {value}")
    logging.info("=" * 40)
    
    # Save the study results with more detailed information
    study_path = '../../data/logs/optuna_study.json'
    with open(study_path, 'w') as f:
        json.dump({
            'best_trial': {
                'number': study.best_trial.number,
                'value': study.best_trial.value,
                'params': study.best_trial.params,
                'datetime': datetime.now().isoformat()
            },
            'all_trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name,
                    'datetime': datetime.now().isoformat()
                }
                for trial in study.trials
            ],
            'optimization_info': {
                'n_trials': N_TRIALS,
                'timesteps_per_trial': 400000,
                'eval_episodes': 20,
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat()
            }
        }, f, indent=4)
    
    logging.info(f"\nResults saved to {study_path}")
    logging.info(f"Full log available at: {log_file}")

if __name__ == "__main__":
    main() 