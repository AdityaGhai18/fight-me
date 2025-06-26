import retro
import time
from gymnasium import Env
from gymnasium.spaces import MultiBinary, Box
import numpy as np
import cv2
import logging
import random

#creating a custon env

class StreetFighter(Env):
    """
    basically retro-gym already has created a base env that give us decent interaction with the game
    we can write custonm environemnts so that we can do things the way we want to
    preprocessing, specifying action and obs space better, setting custom reward functions
    will also use a custom env to write the pvp element so that we can have 2 models playing each other

    honestly we can give a shell of the env and let the students probably define the methods themselves 
    could give some pointers on that
    """ 

    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        
        #start the game, using filtered actions for action space
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', 
                             use_restricted_actions=retro.Actions.FILTERED,
                             render_mode="human")  
        self.previous_score = 0
        self.previous_health = 176
        self.previous_matches_won = 0
        self.previous_enemy_matches_won = 0
        self.score = 0
        self.previous_enemy_health = 176  
        self.previous_frame = None
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            # Also set the seed for the underlying game
            self.game.reset(seed=seed)
        else:
            # If no seed provided, use a random one for variety
            random_seed = random.randint(0, 2**32 - 1)
            super().reset(seed=random_seed)
            self.game.reset(seed=random_seed)
        
        # reset the game to the very first frame 
        obs, info = self.game.reset()
        #take the obs space and preprocess what you get
        obs = self.preprocess(obs)
        self.previous_frame = obs
        #so we store the previous frame when we are going to the next frame
        #do this so we can deal in frame deltas rather than fully new frames every time
        #easier for agent/model to focus on changes
        self.previous_score = info.get('score', 0)
        self.previous_health = info.get('health', 176)
        self.previous_matches_won = info.get('matches_won', 0)
        self.previous_enemy_matches_won = info.get('enemy_matches_won', 0)
        self.score = info.get('score', 0)
        self.previous_enemy_health = info.get('enemy_health', 176)  # Initialize from info
        return obs, info
    
    def preprocess(self, observation): 
        # b/w the frame 
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        # resize for nomarlize/standardize input 
        resize = cv2.resize(gray, (84,84), interpolation=cv2.INTER_CUBIC)
        # add the channels value for 3rd dim
        channels = np.reshape(resize, (84,84,1))
        return channels #just the image of the frame as np array
    
    def step(self, action): 
        obs, _, terminated, truncated, info = self.game.step(action)
        obs = self.preprocess(obs)

        # Calculate frame delta
        frame_delta = obs.astype(np.int16) - self.previous_frame.astype(np.int16)
        self.previous_frame = obs

        # Normalize the delta to be in the [0, 255] range and cast to uint8
        # This is crucial for compatibility with Stable Baselines3's CnnPolicy
        normalized_delta = ((frame_delta + 255) / 2).astype(np.uint8)

        current_health = info.get('health', 176)
        current_score = info.get('score', 0)
        current_matches_won = info.get('matches_won', 0)
        current_enemy_matches_won = info.get('enemy_matches_won', 0)
        current_enemy_health = info.get('enemy_health', 176)

        # Health penalty for our agent
        health_reward = (current_health - self.previous_health) * 5

        # Score reward for hitting enemy
        score_reward = current_score - self.score

        # Enemy health reward (positive for damage dealt)
        enemy_health_reward = (self.previous_enemy_health - current_enemy_health) * 5

        # Match loss penalty (reduced)
        match_loss_penalty = 0
        if current_enemy_matches_won > self.previous_enemy_matches_won:
            match_loss_penalty = -1500

        # Combine rewards
        reward = score_reward + health_reward + enemy_health_reward + match_loss_penalty

        # Update previous values
        self.previous_health = current_health
        self.score = current_score
        self.previous_matches_won = current_matches_won
        self.previous_enemy_matches_won = current_enemy_matches_won
        self.previous_enemy_health = current_enemy_health

        # Add the raw frame to info for visualization
        info['raw_frame'] = self.previous_frame

        return normalized_delta, reward, terminated, truncated, info
    
    def render(self, *args, **kwargs):
        self.game.render()
        
    def close(self):
        try:
            self.game.close()
        except Exception as e:
            logging.warning(f"Error closing environment: {str(e)}")
            # Try to clean up any remaining resources
            try:
                if hasattr(self.game, 'viewer') and self.game.viewer is not None:
                    self.game.viewer = None
            except:
                pass 