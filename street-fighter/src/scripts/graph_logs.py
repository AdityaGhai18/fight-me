import re
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def parse_training_log(log_file_path):
    """Parse training log to extract evaluation data"""
    
    with open(log_file_path, 'r') as f:
        log_content = f.read()
    
    # Debug: Print a sample of the log content
    print("Sample log content:")
    lines = log_content.split('\n')
    for i, line in enumerate(lines):
        if 'Evaluation at step' in line:
            print(f"Line {i}: {line}")
            if i + 4 < len(lines):
                for j in range(1, 5):
                    print(f"Line {i+j}: {lines[i+j]}")
            break
    
    # Find all evaluation sections - use normalized episode rewards
    # Simpler pattern that matches the actual format
    eval_pattern = re.compile(
        r"Evaluation at step (\d+):\s*\n"
        r".*?Mean reward: ([\-\d\.]+) \+/- ([\-\d\.]+)\s*\n"
        r".*?Detailed stats \(5 episodes\):\s*\n"
        r".*?Avg episode reward: ([\-\d\.]+)\s*\n"
        r".*?Min/Max reward: ([\-\d\.]+)/([\-\d\.]+)",
        re.MULTILINE | re.DOTALL
    )
    
    matches = eval_pattern.findall(log_content)
    print(f"Found {len(matches)} evaluation matches")
    
    data = {
        'eval_steps': [],
        'raw_mean_rewards': [],  # Raw values from VecNormalize (not used for plotting)
        'raw_std_rewards': [],   # Raw std from VecNormalize (not used for plotting)
        'normalized_avg_rewards': [],  # This is what we actually want to plot
        'min_rewards': [],
        'max_rewards': [],
        'training_steps': []
    }
    
    for match in matches:
        eval_step = int(match[0])
        raw_mean = float(match[1])      # Raw mean (like 30590.00)
        raw_std = float(match[2])       # Raw std (like 14199.86)
        normalized_avg = float(match[3]) # Normalized avg (like 31.9) - THIS IS WHAT WE WANT
        min_reward = float(match[4])
        max_reward = float(match[5])
        
        # Convert evaluation steps to training steps (4 parallel environments)
        training_step = eval_step * 4
        
        data['eval_steps'].append(eval_step)
        data['raw_mean_rewards'].append(raw_mean)
        data['raw_std_rewards'].append(raw_std)
        data['normalized_avg_rewards'].append(normalized_avg)
        data['min_rewards'].append(min_reward)
        data['max_rewards'].append(max_reward)
        data['training_steps'].append(training_step)
    
    return data

def plot_training_progress(log_file_path, save_plot=True):
    """Plot training progress with error bars"""
    
    data = parse_training_log(log_file_path)
    
    if not data['training_steps']:
        print(f"No evaluation data found in {log_file_path}")
        return
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Normalized average reward with min/max range
    ax1.plot(data['training_steps'], data['normalized_avg_rewards'], 
            'o-', label='Normalized Avg Reward (5 episodes)', color='blue', linewidth=2, markersize=6)
    
    # Add min/max range as shaded area
    ax1.fill_between(data['training_steps'], 
                    data['min_rewards'], 
                    data['max_rewards'], 
                    alpha=0.3, color='red', label='Min/Max Range')
    
    ax1.set_xlabel('Training Timesteps')
    ax1.set_ylabel('Normalized Reward')
    ax1.set_title('Training Progress - Normalized Episode Rewards')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Detailed episode statistics
    ax2.plot(data['training_steps'], data['normalized_avg_rewards'], 
            'o-', label='Normalized Avg Reward', color='green', linewidth=2)
    ax2.plot(data['training_steps'], data['min_rewards'], 
            's-', label='Min Reward', color='red', alpha=0.7, markersize=4)
    ax2.plot(data['training_steps'], data['max_rewards'], 
            '^-', label='Max Reward', color='orange', alpha=0.7, markersize=4)
    
    ax2.set_xlabel('Training Timesteps')
    ax2.set_ylabel('Normalized Reward')
    ax2.set_title('Detailed Episode Statistics (Normalized)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_plot:
        # Extract timestamp from log filename
        log_filename = os.path.basename(log_file_path)
        timestamp = log_filename.replace('training_', '').replace('.log', '')
        plot_filename = f'training_progress_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_filename}")
    
    plt.show()
    
    # Print summary statistics using normalized values
    print(f"\nTraining Summary (Normalized Rewards):")
    print(f"Total evaluations: {len(data['training_steps'])}")
    print(f"Final normalized avg reward: {data['normalized_avg_rewards'][-1]:.2f}")
    print(f"Best normalized avg reward: {max(data['normalized_avg_rewards']):.2f}")
    print(f"Training steps completed: {data['training_steps'][-1]:,}")
    
    # Also show raw vs normalized comparison for debugging
    print(f"\nRaw vs Normalized Comparison (last evaluation):")
    print(f"Raw mean reward: {data['raw_mean_rewards'][-1]:.2f} ± {data['raw_std_rewards'][-1]:.2f}")
    print(f"Normalized avg reward: {data['normalized_avg_rewards'][-1]:.2f}")
    print(f"Min/Max normalized: {data['min_rewards'][-1]:.2f}/{data['max_rewards'][-1]:.2f}")

def find_latest_log(log_dir='../../data/logs/logs3'):
    """Find the most recent training log file"""
    log_files = glob.glob(f"{log_dir}/training_*.log")
    if not log_files:
        print(f"No log files found in {log_dir}")
        return None
    
    # Sort by modification time and get the latest
    latest_log = max(log_files, key=os.path.getmtime)
    return latest_log

if __name__ == "__main__":
    # Find the latest log file
    latest_log = find_latest_log()
    
    if latest_log:
        print(f"Plotting data from: {latest_log}")
        plot_training_progress(latest_log)
    else:
        print("No log files found. Please specify a log file path manually.")
        # You can also manually specify a log file:
        # plot_training_progress('../../data/logs/logs3/training_20250620_132359.log')

