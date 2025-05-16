import gymnasium as gym
import numpy as np
from train import SACAgent
# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        self.state_dim = 67
        self.action_dim = 21
        self.action_space_low = -1.0
        self.action_space_high = 1.0
        self.agent = SACAgent(self.state_dim, self.action_dim, self.action_space_low, self.action_space_high, 256)
        self.agent.load("./sac_humanoid_models")
        self.agent.eval()
    def act(self, observation):
        return self.agent.select_action(observation, evaluate=True)
