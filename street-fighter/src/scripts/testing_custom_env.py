from src.env_iterations.custom_env import StreetFighter
import time
import retro
import numpy as np

#creating the environment from the custom class
env = StreetFighter()

print(env.observation_space.shape)  # should be (84, 84, 1)
print(env.action_space.shape) # should be (12,)

# Reset game to starting state
obs = env.reset()
# Set flag to flase
done = False
for game in range(1): 
    while not done: 
        if done: 
            obs = env.reset()
        env.render()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        time.sleep(0.01)
        if reward > 0: 
            print(reward)

obs = env.reset()