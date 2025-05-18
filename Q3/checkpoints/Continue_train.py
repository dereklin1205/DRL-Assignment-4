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
ALPHA_INITIAL = 0.2 # Initial temperature, can be learned (or fixed if AUTO_TUNE_ALPHA=False)
REPLAY_BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
HIDDEN_DIM = 512
LOG_STD_MIN = -20     # Min log standard deviation for actor
LOG_STD_MAX = 2       # Max log standard deviation for actor
TARGET_ENTROPY = None # If None, will be set to -action_dim
AUTO_TUNE_ALPHA = True # Whether to automatically tune the temperature alpha
def weight_init(m):
	"""Custom weight initialization for better training convergence"""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data, gain=1.0)
		if m.bias is not None:
			nn.init.constant_(m.bias.data, 0.0)
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
    
    def save(self, path):
        """Save replay buffer to a file"""
        buffer_data = np.array(list(self.buffer), dtype=object)
        np.save(f"{path}/replay_buffer.npy", buffer_data)
        print(f"Replay buffer saved to {path}/replay_buffer.npy")
    
    def load(self, path):
        """Load replay buffer from a file"""
        try:
            buffer_data = np.load(f"{path}/replay_buffer.npy", allow_pickle=True)
            self.buffer = deque(list(buffer_data), maxlen=self.buffer.maxlen)
            print(f"Loaded {len(self.buffer)} transitions from {path}/replay_buffer.npy")
            return True
        except FileNotFoundError:
            print(f"No replay buffer found at {path}/replay_buffer.npy")
            return False

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.apply(weight_init) # Apply custom weight initialization
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
        self.apply(weight_init) # Apply custom weight initialization
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
        self.apply(weight_init) # Apply custom weight initialization
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
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0) # Gradient clipping
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
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 1.0) # Gradient clipping
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
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0) # Gradient clipping
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

    def save_models(self, path=".", prefix="sac"):
        """Save all model parameters with an optional prefix"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path}/{prefix}_actor.pth")
        torch.save(self.critic_net.state_dict(), f"{path}/{prefix}_critic.pth")
        torch.save(self.value_net.state_dict(), f"{path}/{prefix}_value.pth")
        if self.auto_tune_alpha:
            # Save log_alpha as it's the parameter being optimized
            torch.save(self.log_alpha, f"{path}/{prefix}_log_alpha.pth")
        print(f"Models saved to {path} with prefix {prefix}")

    def load_models(self, path=".", prefix="sac"):
        """Load all model parameters with an optional prefix"""
        try:
            self.actor.load_state_dict(torch.load(f"{path}/{prefix}_actor.pth", map_location=DEVICE))
            self.critic_net.load_state_dict(torch.load(f"{path}/{prefix}_critic.pth", map_location=DEVICE))
            self.value_net.load_state_dict(torch.load(f"{path}/{prefix}_value.pth", map_location=DEVICE))
            self.target_value_net.load_state_dict(self.value_net.state_dict()) # Sync target V with loaded V
            if self.auto_tune_alpha:
                self.log_alpha = torch.load(f"{path}/{prefix}_log_alpha.pth", map_location=DEVICE)
                self.log_alpha.requires_grad_(True) # Ensure it's a leaf tensor with grad
                self.alpha = self.log_alpha.exp().detach()
                # Re-initialize alpha_optimizer to track the loaded log_alpha
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)

            print(f"Models loaded from {path} with prefix {prefix}")
            return True
        except FileNotFoundError as e:
            print(f"Could not load models: {e}")
            return False


# --- Training Configuration ---
class TrainingConfig:
    def __init__(self):
        self.max_episodes = 10000        # Total number of episodes to train for
        self.max_steps_per_episode = 1000 # DMC tasks typically run for 1000 steps
        self.log_interval = 10           # Log training status every N episodes
        self.save_interval = 200         # Save models every N episodes
        self.warmup_steps = BATCH_SIZE * 10  # Collect some random experiences before starting training
        self.updates_per_step = 1        # Number of agent updates per environment step after warmup
        self.eval_episodes = 10          # Number of episodes to average for evaluation
        self.continue_training = False   # Whether to continue from a previous training run
        self.save_best = True            # Whether to save the best model based on evaluation performance
        self.save_buffer = True          # Whether to save the replay buffer when saving models
        self.checkpoint_dir = "./sac_humanoid_models"  # Directory for saving/loading checkpoints
        self.best_model_dir = "./sac_humanoid_models/best"  # Directory for saving best model
        self.reward_scaling = 5.0        # Factor to scale rewards by
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)


# --- Utility Functions ---
def evaluate_agent(agent, env, config):
    """Evaluate agent over multiple episodes and return average reward"""
    avg_reward = 0
    for i in range(config.eval_episodes):
        eval_state, _ = env.reset()
        episode_reward = 0
        for _ in range(config.max_steps_per_episode):
            eval_action = agent.select_action(eval_state, evaluate=True)
            eval_next_state, eval_reward, eval_term, eval_trunc, _ = env.step(eval_action)
            episode_reward += eval_reward
            eval_state = eval_next_state
            if eval_term or eval_trunc:
                break
        avg_reward += episode_reward
    avg_reward /= config.eval_episodes
    return avg_reward


def save_training_state(agent, buffer, config, episode, total_steps, best_eval_reward):
    """Save training state including agent models, replay buffer, and metadata"""
    # Save metadata
    metadata = {
        "episode": episode,
        "total_steps": total_steps,
        "best_eval_reward": best_eval_reward,
    }
    import json
    with open(f"{config.checkpoint_dir}/metadata.json", "w") as f:
        json.dump(metadata, f)
    
    # Save agent models
    agent.save_models(path=config.checkpoint_dir)
    
    # Save replay buffer if configured
    if config.save_buffer:
        buffer.save(config.checkpoint_dir)


def load_training_state(agent, buffer, config):
    """Load training state including agent models, replay buffer, and metadata"""
    # Load metadata
    import json
    try:
        with open(f"{config.checkpoint_dir}/metadata.json", "r") as f:
            metadata = json.load(f)
        episode = metadata["episode"]
        total_steps = metadata["total_steps"]
        best_eval_reward = metadata["best_eval_reward"]
        
        # Load agent models
        models_loaded = agent.load_models(path=config.checkpoint_dir)
        
        # Load replay buffer if it exists
        buffer_loaded = False
        if config.save_buffer:
            buffer_loaded = buffer.load(config.checkpoint_dir)
        
        if models_loaded:
            print(f"Resuming training from episode {episode+1}, total steps {total_steps}")
            print(f"Best evaluation reward so far: {best_eval_reward:.2f}")
            return episode, total_steps, best_eval_reward, True
        
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Could not load training state: {e}")
    
    # If we're here, loading failed
    print("Starting training from scratch")
    return 0, 0, float('-inf'), False


# --- Main Training Loop ---
def train(config=None):
    if config is None:
        config = TrainingConfig()
        
    print(f"Training on device: {DEVICE}")
    env = make_env()

    # Seeding for reproducibility
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

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

    # Initialize agent and replay buffer
    agent = SACAgent(state_dim, action_dim, action_space_low, action_space_high, HIDDEN_DIM)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    # Initialize training state variables
    start_episode = 0
    total_steps = 0
    best_eval_reward = float('-inf')
    
    # Load previous training state if continuing training
    if config.continue_training:
        start_episode, total_steps, best_eval_reward, success = load_training_state(agent, replay_buffer, config)
        if not success and config.continue_training:
            print("Warning: Failed to load previous training state but continue_training=True")
            print("Proceeding with new training run")

    current_alpha = agent.alpha.item() # Initialize current alpha for logging
    v_loss, c_loss, a_loss = 0.0, 0.0, 0.0 # Initialize losses for logging

    for episode in range(start_episode + 1, config.max_episodes + 1):
        seed = np.random.randint(0, 1000000) # Random seed for each episode
        state, _ = env.reset(seed=seed)
        episode_reward = 0
        episode_steps = 0

        for step_in_episode in range(config.max_steps_per_episode):
            if total_steps < config.warmup_steps and len(replay_buffer) < BATCH_SIZE:
                action = env.action_space.sample() # Sample random action during warmup
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated # In DMC, terminated is the primary 'done' signal
            shaped_reward = reward * config.reward_scaling # Shape the reward
            replay_buffer.push(state, action, shaped_reward, next_state, float(done)) # Store done as float (0.0 or 1.0)
            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_steps += 1

            if len(replay_buffer) > BATCH_SIZE and total_steps >= config.warmup_steps:
                for _ in range(config.updates_per_step): # Perform one or more updates
                    v_loss, c_loss, a_loss, current_alpha = agent.update(replay_buffer, BATCH_SIZE)

            if done:
                break
                
        # Log episode statistics
        print(f"Episode: {episode}, Total Steps: {total_steps}, Episode Reward: {episode_reward:.2f}, Episode Steps: {episode_steps}, Alpha: {current_alpha:.3f},V_LOSS: {v_loss:.3f}, C_LOSS: {c_loss:.3f}, A_LOSS: {a_loss:.3f}")
        
        # Periodic evaluation and saving
        if episode % config.log_interval == 0:
            # Evaluate agent performance
            avg_reward_eval = evaluate_agent(agent, env, config)
            
            print(f"--------------------------------------------------------")
            print(f"Episode: {episode}, Total Steps: {total_steps}, Train Reward: {episode_reward:.2f}, Avg Eval Reward: {avg_reward_eval:.2f}, Ep Steps: {episode_steps}")
            print(f"--------------------------------------------------------")
            
            # Save best model if performance improves
            if config.save_best and avg_reward_eval > best_eval_reward:
                best_eval_reward = avg_reward_eval
                agent.save_models(path=config.best_model_dir, prefix="best")
                print(f"New best model saved with reward: {best_eval_reward:.2f}")
        
        # Regular checkpoint saving
        if episode % config.save_interval == 0 and total_steps > config.warmup_steps:
            save_training_state(agent, replay_buffer, config, episode, total_steps, best_eval_reward)
            print(f"Checkpoint saved at episode {episode}")

    # Final save at the end of training
    save_training_state(agent, replay_buffer, config, config.max_episodes, total_steps, best_eval_reward)
    print("Training complete!")


def evaluate(model_dir="./sac_humanoid_models/best", num_episodes=10, render=False):
    """Evaluate a trained agent for a number of episodes"""
    print(f"Evaluating agent from {model_dir}")
    env = make_env()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high
    
    # Initialize agent
    agent = SACAgent(state_dim, action_dim, action_space_low, action_space_high, HIDDEN_DIM)
    
    # Try to load the best model first, fall back to regular checkpoint if not found
    try:
        success = agent.load_models(path=model_dir, prefix="best")
        if not success:
            success = agent.load_models(path=model_dir)
        if not success:
            print(f"No models found in {model_dir}")
            return
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Set agent to evaluation mode
    agent.actor.eval()
    
    # Evaluate for several episodes
    total_rewards = []
    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 1000:  # Max steps to avoid infinite episodes
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            done = terminated or truncated
            step += 1
            
        total_rewards.append(episode_reward)
        print(f"Episode {episode}/{num_episodes}: Reward = {episode_reward:.2f}, Steps = {step}")
    
    # Print statistics
    avg_reward = sum(total_rewards) / len(total_rewards)
    std_reward = np.std(total_rewards)
    min_reward = min(total_rewards)
    max_reward = max(total_rewards)
    
    print("\n--- Evaluation Results ---")
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min/Max Reward: {min_reward:.2f}/{max_reward:.2f}")
    print("-------------------------")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train or evaluate SAC agent on DMC humanoid-walk task")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Train the agent")
    mode_group.add_argument("--evaluate", action="store_true", help="Evaluate the agent")
    
    # Training arguments
    parser.add_argument("--continue_training", action="store_true", help="Continue from previous checkpoint")
    parser.add_argument("--model_dir", type=str, default="./sac_humanoid_model_train_new", 
                        help="Directory for saving/loading models")
    parser.add_argument("--best_model_dir", type=str, default=None,
                        help="Directory for saving best models (defaults to model_dir/best)")
    parser.add_argument("--max_episodes", type=int, default=10000, 
                        help="Maximum number of episodes for training")
    parser.add_argument("--save_interval", type=int, default=200,
                        help="Save checkpoint every N episodes")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log and evaluate every N episodes")
    parser.add_argument("--reward_scaling", type=float, default=5.0,
                        help="Factor to scale rewards by")
    
    # Evaluation arguments
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
                        
    args = parser.parse_args()
    
    # Handle model directories
    model_dir = args.model_dir
    best_model_dir = args.best_model_dir if args.best_model_dir else os.path.join(model_dir, "best")
    
    # Create necessary directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    
    if args.train:
        # Configure training
        config = TrainingConfig()
        config.continue_training = args.continue_training
        config.checkpoint_dir = model_dir
        config.best_model_dir = best_model_dir
        config.max_episodes = args.max_episodes
        config.save_interval = args.save_interval
        config.log_interval = args.log_interval
        config.reward_scaling = args.reward_scaling
        
        # Start training
        print(f"Training agent with config:")
        print(f"  Model directory: {model_dir}")
        print(f"  Best model directory: {best_model_dir}")
        print(f"  Continue training: {args.continue_training}")
        print(f"  Max episodes: {args.max_episodes}")
        print(f"  Reward scaling: {args.reward_scaling}")
        train(config=config)
        
    elif args.evaluate:
        # Start evaluation
        evaluate(model_dir=model_dir, num_episodes=args.eval_episodes)