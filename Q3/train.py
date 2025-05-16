import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
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
# from dm_env import specs # For type hinting if desired

# --- Environment Setup ---
def make_env():
    env_name = "humanoid-walk"
    env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    return env

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
LR_ACTOR = 3e-4       # Learning rate for actor
LR_CRITIC = 3e-4      # Learning rate for critics
LR_VALUE = 3e-4       # Learning rate for value network
LR_ALPHA = 3e-4       # Learning rate for temperature alpha
GAMMA = 0.99          # Discount factor
TAU = 0.005           # Target network soft update rate
ALPHA_INITIAL = 1  # Initial temperature, can be learned (or fixed if AUTO_TUNE_ALPHA=False)
REPLAY_BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
HIDDEN_DIM = 512
LOG_STD_MIN = -20     # Min log standard deviation for actor
LOG_STD_MAX = 2       # Max log standard deviation for actor
TARGET_ENTROPY = None # If None, will be set to -action_dim
AUTO_TUNE_ALPHA = True # Whether to automatically tune the temperature alpha

# --- Networks ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.array(state)).to(DEVICE),
            torch.FloatTensor(np.array(action)).to(DEVICE),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(np.array(next_state)).to(DEVICE),
            torch.FloatTensor(np.array(done)).unsqueeze(1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CriticNetwork(nn.Module): # Q-Network
    def __init__(self, state_dim, action_dim, hidden_dim = 256):
        super(CriticNetwork, self).__init__()
        # Q1 architecture
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4_q1 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4_q2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.fc1_q1(sa))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = F.relu(self.fc3_q1(q1))
        q1 = self.fc4_q1(q1)
        

        q2 = F.relu(self.fc1_q2(sa))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = F.relu(self.fc3_q2(q2))
        q2 = self.fc4_q2(q2)
        return q1, q2

    def Q1(self, state, action): # Utility to get Q1 value, useful for actor update
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1_q1(sa))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = F.relu(self.fc3_q1(q1))
        q1 = self.fc4_q1(q1)
        return q1

class ActorNetwork(nn.Module): # Policy Network
    def __init__(self, state_dim, action_dim, hidden_dim, action_space_low, action_space_high):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc = nn.Linear(hidden_dim, action_dim)
        self.log_std_fc = nn.Linear(hidden_dim, action_dim)

        # Action scaling
        self.action_scale = torch.FloatTensor((action_space_high - action_space_low) / 2.0).to(DEVICE)
        self.action_bias = torch.FloatTensor((action_space_high + action_space_low) / 2.0).to(DEVICE)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean_fc(x)
        log_std = self.log_std_fc(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state, reparameterize=True):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        if reparameterize:
            x_t = normal.rsample()  # Reparameterization trick (mean + std * N(0,1))
        else:
            x_t = normal.sample()

        y_t = torch.tanh(x_t)  # Squash to [-1, 1] range
        action = y_t * self.action_scale + self.action_bias # Scale to action space

        # Calculate log probability with correction for Tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6) # 1e-6 for numerical stability
        log_prob = log_prob.sum(1, keepdim=True)

        # For evaluation, also return the deterministic (mean) action after squashing and scaling
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

# --- SAC Agent ---
class SACAgent:
    def __init__(self, state_dim, action_dim, action_space_low, action_space_high, hidden_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor Network
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim, action_space_low, action_space_high).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # Value Network
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(DEVICE)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=LR_VALUE)
        self.target_value_net = ValueNetwork(state_dim, hidden_dim).to(DEVICE)
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        for p in self.target_value_net.parameters():
            p.requires_grad = False # Target network parameters are not trained directly via an optimizer

        # Critic Networks (Twin Q-networks)
        self.critic_net = CriticNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=LR_CRITIC)

        # Temperature (alpha)
        self.auto_tune_alpha = AUTO_TUNE_ALPHA
        if self.auto_tune_alpha:
            global TARGET_ENTROPY # Use global TARGET_ENTROPY if it's set outside
            if TARGET_ENTROPY is None:
                self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(DEVICE)).item() # Heuristic: -dim(A)
            else:
                self.target_entropy = TARGET_ENTROPY
            
            self.log_alpha = torch.tensor(np.log(1), dtype=torch.float32, requires_grad=True, device=DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)
            self.alpha = self.log_alpha.exp().detach() # Detach alpha for use in losses where its grad isn't needed for itself
        else:
            self.alpha = torch.tensor(ALPHA_INITIAL, dtype=torch.float32, device=DEVICE)


    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).to(DEVICE).unsqueeze(0)
        if evaluate: # During evaluation, use the mean action (deterministic)
            with torch.no_grad():
                _, _, action = self.actor.sample(state_tensor, reparameterize=False)
        else: # During training, sample stochastically
            with torch.no_grad():
                action, _, _ = self.actor.sample(state_tensor, reparameterize=True) # reparam=False as we don't need grad here
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # --- Update Value Network ---
        # Sample new actions from current policy for these states (for V target)
        with torch.no_grad(): # Actions for V target don't need grads propagating back to actor here
            new_actions_for_v, log_probs_for_v, _ = self.actor.sample(states)
            q1_new_policy, q2_new_policy = self.critic_net(states, new_actions_for_v)
            q_new_policy = torch.min(q1_new_policy, q2_new_policy)
        # V_target = Q_new_policy - alpha * log_probs
        value_target = q_new_policy - self.alpha * log_probs_for_v # Use current alpha
        value_predicted = self.value_net(states)
        value_loss = F.mse_loss(value_predicted, value_target.detach()) # Detach target

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # --- Update Critic Networks (Q-networks) ---
        with torch.no_grad(): # Targets should not contribute to gradient
            next_value_from_target_v = self.target_value_net(next_states)
            # Q_target = r + gamma * (1-done) * V_target(s')
            q_target = rewards + GAMMA * (1.0 - dones) * next_value_from_target_v

        q1_pred, q2_pred = self.critic_net(states, actions)
        critic_loss_q1 = F.mse_loss(q1_pred, q_target)
        critic_loss_q2 = F.mse_loss(q2_pred, q_target)
        critic_loss = critic_loss_q1 + critic_loss_q2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor Network (Policy) ---
        # Resample actions with reparameterization for policy gradient
        pi_actions, log_probs, _ = self.actor.sample(states, reparameterize=True)
        q1_pi, q2_pi = self.critic_net(states, pi_actions)
        q_pi = torch.min(q1_pi, q2_pi) # Use min of Q critics for actor update

        # Policy_loss = (alpha * log_probs - Q_for_actor).mean()
        actor_loss = (self.alpha.detach() * log_probs - q_pi).mean() # Detach alpha here, alpha grad comes from alpha_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update Temperature (alpha) ---
        if self.auto_tune_alpha:
            # log_probs are from the reparameterized sample used for actor update
            alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
            # Alternative formulation: alpha_loss = (self.log_alpha * (-log_probs - self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach() # Update alpha value for next iteration's losses

        # --- Soft Update Target Value Network ---
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        return value_loss.item(), critic_loss.item(), actor_loss.item(), self.alpha.item()

    def save_models(self, path="."):
        torch.save(self.actor.state_dict(), f"{path}/sac_actor.pth")
        torch.save(self.critic_net.state_dict(), f"{path}/sac_critic.pth")
        torch.save(self.value_net.state_dict(), f"{path}/sac_value.pth")
        if self.auto_tune_alpha:
            # Save log_alpha as it's the parameter being optimized
            torch.save(self.log_alpha, f"{path}/sac_log_alpha.pth")
        print(f"Models saved to {path}")

    def load_models(self, path="."):
        self.actor.load_state_dict(torch.load(f"{path}/sac_actor.pth", map_location=DEVICE))
        self.critic_net.load_state_dict(torch.load(f"{path}/sac_critic.pth", map_location=DEVICE))
        self.value_net.load_state_dict(torch.load(f"{path}/sac_value.pth", map_location=DEVICE))
        self.target_value_net.load_state_dict(self.value_net.state_dict()) # Sync target V with loaded V
        if self.auto_tune_alpha:
            self.log_alpha = torch.load(f"{path}/sac_log_alpha.pth", map_location=DEVICE)
            self.log_alpha.requires_grad_(True) # Ensure it's a leaf tensor with grad
            self.alpha = self.log_alpha.exp().detach()
            # Re-initialize alpha_optimizer if it was saved/loaded or ensure it tracks the loaded log_alpha
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)

        print(f"Models loaded from {path}")
        self.actor.eval() # Set to eval mode if loading for inference
        self.critic_net.eval()
        self.value_net.eval()


# --- Main Training Loop ---
def train():
    print(f"Training on device: {DEVICE}")
    env = make_env()

    # Seeding for reproducibility
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    # Note: DMC envs are seeded at creation via task_kwargs.
    # For action space sampling during warmup if it's a Box space:
    # env.action_space.seed(RANDOM_SEED) # If using a gym.spaces.Box like wrapper

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high

    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Action space low: {action_space_low[0]:.2f}, high: {action_space_high[0]:.2f} (showing first dim)")

    global TARGET_ENTROPY # Allow modification if it's None
    if TARGET_ENTROPY is None and AUTO_TUNE_ALPHA:
        TARGET_ENTROPY = -float(action_dim) # Heuristic for continuous action spaces
        print(f"Target entropy automatically set to: {TARGET_ENTROPY:.2f}")

    agent = SACAgent(state_dim, action_dim, action_space_low, action_space_high, HIDDEN_DIM)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    max_episodes = 10000  # Total number of episodes to train for
    max_steps_per_episode = 1000 # DMC tasks typically run for 1000 steps
    log_interval = 10     # Log training status every N episodes
    save_interval = 200   # Save models every N episodes
    warmup_steps = BATCH_SIZE * 10 # Collect some random experiences before starting training
    updates_per_step = 1 # Number of agent updates per environment step after warmup
    current_alpha = agent.alpha.item() # Initialize current alpha for logging
    v_loss, c_loss, a_loss = 0.0, 0.0, 0.0 # Initialize losses for logging
    total_steps = 0
    for episode in range(1, max_episodes + 1):
        seed = np.random.randint(0, 1000000) # Random seed for each episode
        state, _ = env.reset(seed = seed)
        # print(f"Seed = {seed}")
        episode_reward = 0
        episode_steps = 0

        for step_in_episode in range(max_steps_per_episode):
            if total_steps < warmup_steps:
                action = env.action_space.sample() # Sample random action
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated # In DMC, terminated is the primary 'done' signal
            shaped_reward = reward*5.0 # Modify this if you want to shape the reward (e.g. scaled, clipped, etc.)
            replay_buffer.push(state, action, shaped_reward, next_state, float(done)) # Store done as float (0.0 or 1.0)
            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_steps += 1

            if len(replay_buffer) > BATCH_SIZE and total_steps >= warmup_steps:
                for _ in range(updates_per_step): # Perform one or more updates
                    v_loss, c_loss, a_loss, current_alpha = agent.update(replay_buffer, BATCH_SIZE)

                # if total_steps % (updates_per_step * 200) == 0: # Log losses occasionally (e.g. every 200 updates)
                #     print(f"Ep: {episode}, TotSteps: {total_steps}, V_Loss: {v_loss:.3f}, "
                #           f"C_Loss: {c_loss:.3f}, A_Loss: {a_loss:.3f}, Alpha: {current_alpha:.3f}")

            if done:
                break
        print(f"Episode: {episode}, Total Steps: {total_steps}, Episode Reward: {episode_reward:.2f}, Episode Steps: {episode_steps}, Alpha: {current_alpha:.3f},V_LOSS: {v_loss:.3f}, C_LOSS: {c_loss:.3f}, A_LOSS: {a_loss:.3f}")
        
        if episode % log_interval == 0:
            avg_reward_eval = 0
            eval_episodes = 3 # Number of episodes to average for evaluation
            for _ in range(eval_episodes):
                eval_state, _ = env.reset()
                eval_episode_reward = 0
                for _ in range(max_steps_per_episode):
                    eval_action = agent.select_action(eval_state, evaluate=True)
                    eval_next_state, eval_reward, eval_term, eval_trunc, _ = env.step(eval_action)
                    eval_episode_reward += eval_reward
                    eval_state = eval_next_state
                    if eval_term or eval_trunc:
                        break
                avg_reward_eval += eval_episode_reward
            avg_reward_eval /= eval_episodes

            print(f"--------------------------------------------------------")
            print(f"Episode: {episode}, Total Steps: {total_steps}, Train Reward: {episode_reward:.2f}, Avg Eval Reward: {avg_reward_eval:.2f}, Ep Steps: {episode_steps}")
            print(f"--------------------------------------------------------")


        if episode % save_interval == 0 and total_steps > warmup_steps:
            agent.save_models(path="./sac_humanoid_models")

    # env.close() # If your env wrapper has a close method, call it here. DMC usually doesn't require explicit close.

if __name__ == "__main__":
    import os
    if not os.path.exists("./sac_humanoid_models"):
        os.makedirs("./sac_humanoid_models")
    print("123")
    train()