# fight-me
Applying ML approaches to fighting games

This repo contains reinforcement learning agents trained to play Street Fighter II using Stable Baselines3 and Stable Retro. 
I will be updating this with newly updated environements, different strategies and possibly a VLA approach coming up!

## Project Structure

```
fight/
├── street-fighter/          # Main Street Fighter AI project
│   ├── src/
│   │   ├── env_iterations/  # Custom environment implementations
│   │   └── scripts/         # Training and evaluation scripts
│   ├── data/
│   │   ├── models/          # Trained models and checkpoints
│   │   └── logs/            # Training logs, Optuna logs, eval results
│   └── roms/                # Game ROMs (not included in repo)
├── requirements.txt         # Python dependencies
└── .gitignore               # Files and folders excluded from version control
```

ROM files not included, Libraries were built from source and not included

## Quick Start

## Environment Versions
- **custom_env.py**: Basic implementation
- **custom_env2.py**: Improved rewards
- **custom_env3.py**: Frame delta observations + combo action space + RAM hacked feature vector included in obs (soon!)

## Training Scripts
- `training_ppo.py`: Basic PPO training
- `train_with_optimized_params.py`: Training with optimized hyperparameters from optuna runs
- `optimize_ppo.py`: Optuna hyperparameter search - still finding perfect search range for each parameter
- `evaluate_spike_params.py`: Evaluate specific parameter sets manually watching gameplay
- `play_model.py`: Run a play round using a trained model
- `test_rewards.py`, `random-actions.py`, `testing_custom_env.py`: Testing and debugging

## Latest Performance
Beats quite a few of the initial bots, here is a link to the (best latest run)[https://youtu.be/9xbVs3dH0Nk]


