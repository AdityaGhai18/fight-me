import retro
from stable_baselines3 import PPO
import cv2
import numpy as np
from collections import deque

def preprocess_observation(obs):
    """Convert observation to match training format"""
    # Convert to grayscale
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
    return resized

class FrameStack:
    def __init__(self, n_frames=4):
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
    
    def add_frame(self, frame):
        self.frames.append(frame)
        while len(self.frames) < self.n_frames:
            self.frames.append(frame)  # Duplicate first frame if not enough frames
    
    def get_stacked_frames(self):
        return np.stack(self.frames, axis=0)  # Shape: (4, 84, 84)

def play_model(model_path):
    # Create the environment
    env = retro.make('StreetFighterIISpecialChampionEdition-Genesis')
    frame_stack = FrameStack(n_frames=4)
    
    # Load the model
    model = PPO.load(model_path)
    
    # Reset the environment
    obs, _ = env.reset()
    obs = preprocess_observation(obs)
    frame_stack.add_frame(obs)
    stacked_obs = frame_stack.get_stacked_frames()
    
    # Play the game
    while True:
        action, _ = model.predict(stacked_obs)
        obs, _, terminated, truncated, _ = env.step(action)
        obs = preprocess_observation(obs)
        frame_stack.add_frame(obs)
        stacked_obs = frame_stack.get_stacked_frames()
        if terminated or truncated:
            obs, _ = env.reset()
            obs = preprocess_observation(obs)
            frame_stack.add_frame(obs)
            stacked_obs = frame_stack.get_stacked_frames()

if __name__ == "__main__":
    # Choose one of these model paths:
    model_path = "../../data/models/best_model3/latest_best_model.zip"  # Latest best model
    # model_path = "../../data/models/checkpoints3/model_1000000.zip"   # Latest checkpoint
    # model_path = "../../data/models/ppo_streetfighter.zip"            # Basic trained model
    
    print(f"Loading model from: {model_path}")
    play_model(model_path)
