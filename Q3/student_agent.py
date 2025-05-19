import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import time
import argparse
import importlib
import numpy as np
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env
from train import SACAgent
# Do not modify the input of the 'act' function and the '__init__' function. 
RANDOM_SEED = 42
def make_env():
    env_name = "humanoid-walk"
    env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    return env

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        env = make_env()

    # Seeding for reproducibility
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        # Note: DMC envs are seeded at creation via task_kwargs.
        # For action space sampling during warmup if it's a Box space:
        # env.action_space.seed(RANDOM_SEED) # If using a gym.spaces.Box like wrapper

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_space_low = env.action_space.low
        self.action_space_high = env.action_space.high
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        self.agent = SACAgent(self.state_dim, self.action_dim, self.action_space_low, self.action_space_high, 512)
        self.agent.load_models(path = "./checkpoints/sac_humanoid_model_train_new/best", prefix="best")
        self.agent.actor.eval() # Set to eval mode if loading for inference
        self.agent.critic_net.eval()
        self.agent.value_net.eval()


    def act(self, observation):
        return self.agent.select_action(observation, evaluate=True)
if __name__ == "__main__":
    env = make_env()
    agent = Agent()
    seed = np.random.randint(0, 1000000)
    obs, _ = env.reset(seed=seed) # Reset the environment and get the initial observation
    done = False
    total_reward = 0
    # print(obs[0].shape)
    while not done :
        
        action = agent.act(obs)
        obs, reward, done,truncated, info = env.step(action)
        env.render()
        done = done or truncated
        # time.sleep(0.01)  # Add a small delay to visualize the rendering
        total_reward += reward
    # Close the environment after the loop
    print(f"Total reward: {total_reward}")
    env.close()
