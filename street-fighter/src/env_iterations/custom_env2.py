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
                             render_mode='human')  # Enable human rendering mode
        self.previous_score = 0
        self.previous_health = 176
        self.previous_matches_won = 0
        self.previous_enemy_matches_won = 0
    
    def reset(self, seed=None, options=None):
        # reset the game to the very first frame 
        obs, info = self.game.reset()
        #take the obs space and preprocess what you get
        obs = self.preprocess(obs)
        #so we store the previous frame when we are going to the next frame
        #do this so we can deal in frame deltas rather than fully new frames every time
        #easier for agent/model to focus on changes
        self.previous_score = info.get('score', 0)
        self.previous_health = info.get('health', 176)
        self.previous_matches_won = info.get('matches_won', 0)
        self.previous_enemy_matches_won = info.get('enemy_matches_won', 0)
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


        current_score = info.get('score', 0)
        score_delta = current_score - self.previous_score
        self.previous_score = current_score
        score_norm = 50000  
        reward = score_delta / score_norm

        # Penalize health lost
        current_health = info.get('health', 176)
        health_lost = self.previous_health - current_health
        health_norm = 176
        reward -= (health_lost / health_norm)
        self.previous_health = current_health


        matches_won_bonus = 0.5 * info.get('matches_won', 0)
        reward += matches_won_bonus

        # Reward clipping
        reward = np.clip(reward, -10, 10)

        #print(f"score_delta: {score_delta}, health_lost: {health_lost}, matches_won: {info.get('matches_won', 0)}, reward: {reward}, info: {info}")

        print(reward)
        print(info)
        return obs, reward, terminated, truncated, info
    
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