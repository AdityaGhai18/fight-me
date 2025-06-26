import numpy as np
from src.env_iterations.custom_env3 import StreetFighter
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import time
import cv2

# Create the environment

def make_env():
    return StreetFighter()

# Wrap with DummyVecEnv and VecNormalize (like in training)
env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=False, norm_reward=True)

n_episodes = 10
for ep in range(n_episodes):
    obs = env.reset()
    done = False
    total_norm_reward = 0
    total_raw_reward = 0
    step_count = 0
    print(f"\n=== Episode {ep+1}/{n_episodes} ===")
    while not done:
        action = [env.action_space.sample()]
        obs, norm_reward, done, info = env.step(action)
        
        # Visualize the frame delta
        raw_frame = info[0].get('raw_frame')
        if raw_frame is not None:
            frame_delta = obs[0]
            # Normalize delta for visualization (from [-255, 255] to [0, 255])
            vis_delta = ((frame_delta.astype(np.float32) + 255) / 2).astype(np.uint8)

            # Resize for better viewing
            vis_raw = cv2.resize(raw_frame, (300, 300), interpolation=cv2.INTER_NEAREST)
            vis_delta = cv2.resize(vis_delta, (300, 300), interpolation=cv2.INTER_NEAREST)
            
            # Add labels to the frames
            cv2.putText(vis_raw, 'Raw Frame', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_delta, 'Frame Delta', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Combine frames and show
            combined_view = np.hstack([vis_raw, vis_delta])
            cv2.imshow('Street Fighter - Raw vs. Delta', combined_view)
            cv2.waitKey(1) # Refresh window
            time.sleep(0.01)

        raw_reward = env.get_original_reward()
        total_norm_reward += norm_reward[0]
        total_raw_reward += raw_reward[0]
        step_count += 1
    print(f"Episode {ep+1} finished. Total normalized reward: {total_norm_reward:.2f}, Total raw reward: {total_raw_reward:.2f}")

cv2.destroyAllWindows()
env.close()

