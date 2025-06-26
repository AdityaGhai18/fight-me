import retro
import time
from gymnasium import Env
from gymnasium.spaces import MultiBinary, Box
import numpy as np
import cv2
import logging

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
        # action space and observation space 
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        
        #start the game, using filtered actions for action space
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', 
                             use_restricted_actions=retro.Actions.FILTERED,
                             render_mode=None)  # Disable rendering during training
    
    def reset(self, seed=None, options=None):
        # reset the game to the very first frame 
        obs, info = self.game.reset()
        #take the obs space and preprocess what you get
        obs = self.preprocess(obs)
        #so we store the previous frame when we are going to the next frame
        #do this so we can deal in frame deltas rather than fully new frames every time
        #easier for agent/model to focus on changes
        self.previous_frame = obs
        #score delta attribute
        self.score = 0 
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
        obs, reward, terminated, truncated, info = self.game.step(action)

        obs = self.preprocess(obs)
        
        # calculating the frame delta 
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs 
        
        # Get player and opponent health from info
        player_health = info.get('health', 0)
        opponent_health = info.get('enemy_health', 0)

        # Calculate reward based on absolute health values
        # Normalize health values to be between 0 and 1
        max_health = 176  # Maximum health in Street Fighter
        player_health_norm = player_health / max_health
        opponent_health_norm = opponent_health / max_health
        
        # Reward for having more health than opponent
        reward = (player_health_norm - opponent_health_norm) * 10

        # Add rewards for wins and losses - overall we need to make sure its winning based despite health being the objective 
        if info.get('matches_won', 0) > getattr(self, 'previous_matches_won', 0):
            reward += 10
        if info.get('enemy_matches_won', 0) > getattr(self, 'previous_enemy_matches_won', 0):
            reward -= 10
            
        # Update previous match values
        self.previous_matches_won = info.get('matches_won', 0)
        self.previous_enemy_matches_won = info.get('enemy_matches_won', 0)

        # Big reward for reducing opponent health to 0
        if opponent_health == 0:
            reward += 50
        # Big negative reward for player health reaching 0
        if player_health == 0:
            reward -= 50

        return frame_delta, reward, terminated, truncated, info
    
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
