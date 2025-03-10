import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from custom_env import TrainingTaxiEnv  # Your environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")
# --- Global Definitions ---
stations = {0: 'R', 1: 'G', 2: 'Y', 3: 'B'}

# We will still use these global memory variables to help our policy "remember" events during an episode.
# (Make sure they are reset at the start of each episode.)
pickup = False
visited = [0, 0, 0, 0]
destination = None

# --- State Vector Conversion ---
def get_state_vector(obs, pickup=False, visited=[0, 0, 0, 0], destination=None):
    """
    Converts the observation into a 25-dimensional vector.
    The observation (obs) is assumed to be a tuple with 16 elements:
      (taxi_row, taxi_col,
       Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol,
       obstacle_north, obstacle_south, obstacle_east, obstacle_west,
       passenger_look, destination_look)
    
    The state vector contains:
      - Taxi position (2 dims)
      - Pickup flag (1 dim)
      - Visited stations (4 dims)
      - Destination one-hot (4 dims; if destination is None, this is all zeros)
      - Relative positions for each station (8 dims: dr & dc for R, G, Y, B)
      - Obstacles (4 dims)
      - Passenger look and destination look (2 dims)
    Total dims: 2 + 1 + 4 + 4 + 8 + 4 + 2 = 25.
    """
    taxi_row, taxi_col, Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol, \
        obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
        passenger_look, destination_look = obs

    # Compute relative positions (station minus taxi)
    drR, dcR = Rrow - taxi_row, Rcol - taxi_col
    drG, dcG = Grow - taxi_row, Gcol - taxi_col
    drY, dcY = Yrow - taxi_row, Ycol - taxi_col
    drB, dcB = Brow - taxi_row, Bcol - taxi_col

    pickup_val = 1.0 if pickup else 0.0
    visited_arr = np.array(visited, dtype=np.float32)

    # One-hot encoding for destination (if known)
    dest_vec = np.zeros(4, dtype=np.float32)
    if destination is not None:
        for key, val in stations.items():
            if val == destination:
                dest_vec[key] = 1.0
                break

    obstacles = np.array([obstacle_north, obstacle_south, obstacle_east, obstacle_west], dtype=np.float32)
    passenger_val = float(passenger_look)
    dest_look_val = float(destination_look)

    state_vector = np.concatenate((
        np.array([taxi_row, taxi_col], dtype=np.float32),
        np.array([pickup_val], dtype=np.float32),
        visited_arr,
        dest_vec,
        np.array([drR, dcR, drG, dcG, drY, dcY, drB, dcB], dtype=np.float32),
        obstacles,
        np.array([passenger_val, dest_look_val], dtype=np.float32)
    ))
    return state_vector

# --- Policy Network Definition ---
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return torch.softmax(logits, dim=-1)  # outputs a probability distribution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 25
output_dim = 6  # Actions: South, North, East, West, Pickup, Dropoff
policy_net = PolicyNetwork(input_dim, output_dim).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

def locate_destination(stations_offset):
    """
    Given a list of station offsets [(dr, dc), ...] for R, G, Y, B,
    returns the index of the station that is closest (by Manhattan distance).
    """
    distances = [abs(dr) + abs(dc) for dr, dc in stations_offset]
    return distances.index(min(distances))

# --- Policy Gradient (REINFORCE) Training ---
def train_policy_gradient(total_episodes=2000, gamma=0.99):
    rewards_per_episode = []
    avg_rewards_100 = []
    losses_per_episode = []
    rewards_buffer = deque(maxlen=100)
    
    global pickup, visited, destination
    for episode in range(total_episodes):
        # Initialize/reset environment and memory for each episode.
        env = TrainingTaxiEnv(n=5, max_fuel=5000, obstacle_prob=0)
        obs, _ = env.reset()
        pickup_flag = False
        visited = [0, 0, 0, 0]
        destination = None
        
        state_vec = get_state_vector(obs, pickup_flag, visited, destination)
        done = False
        
        log_probs = []  # Log probabilities of chosen actions
        rewards = []    # Rewards obtained per step
        episode_reward = 0.0
        
        prev_passenger_look = obs[-2]
        
        while not done:
            state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
            # Obtain action probabilities from policy network
            action_probs = policy_net(state_tensor)
            # Sample an action from the distribution
            action = int(torch.multinomial(action_probs, num_samples=1).item())
            # Store log probability of this action
            log_prob = torch.log(action_probs[0, action])
            log_probs.append(log_prob)
            
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # --- Reward Shaping and State Memory Updates ---
            # If action 4 (Pickup) is taken and passenger look changes, mark pickup as done.
            current_passenger_look = next_obs[-2]
            if action == 4 and (current_passenger_look != prev_passenger_look) and not pickup_flag:
                pickup_flag = True
                reward += 100  # bonus for successful pickup
                
            # Update destination when destination look is active
            if next_obs[-1]:
                taxi_row, taxi_col = next_obs[0], next_obs[1]
                offsets = [
                    (next_obs[2] - taxi_row, next_obs[3] - taxi_col),  # R
                    (next_obs[4] - taxi_row, next_obs[5] - taxi_col),  # G
                    (next_obs[6] - taxi_row, next_obs[7] - taxi_col),  # Y
                    (next_obs[8] - taxi_row, next_obs[9] - taxi_col)   # B
                ]
                dest_idx = locate_destination(offsets)
                destination = stations[dest_idx]
            
            # Update visited stations if taxi's position equals station coordinates
            taxi_row, taxi_col = next_obs[0], next_obs[1]
            if taxi_row == next_obs[2] and taxi_col == next_obs[3]:
                visited[0] = 1
            if taxi_row == next_obs[4] and taxi_col == next_obs[5]:
                visited[1] = 1
            if taxi_row == next_obs[6] and taxi_col == next_obs[7]:
                visited[2] = 1
            if taxi_row == next_obs[8] and taxi_col == next_obs[9]:
                visited[3] = 1
            
            rewards.append(reward)
            episode_reward += reward
            prev_passenger_look = current_passenger_look
            
            # Update state vector for next step
            state_vec = get_state_vector(next_obs, pickup_flag, visited, destination)
        
        # Compute discounted returns for the episode.
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        # Normalize returns to reduce variance (optional but recommended)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute the policy gradient loss (REINFORCE loss)
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss -= log_prob * G
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        rewards_per_episode.append(episode_reward)
        rewards_buffer.append(episode_reward)
        avg_rewards_100.append(np.mean(rewards_buffer))
        losses_per_episode.append(loss.item())
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{total_episodes} | Avg Reward: {avg_rewards_100[-1]:.2f} | Loss: {loss.item():.4f}")
    
    # Plot training metrics
    plt.figure(figsize=(14,6))
    plt.subplot(221)
    plt.plot(rewards_per_episode, label="Episode Reward")
    plt.plot(avg_rewards_100, label="Avg Reward (100 eps)", color='red')
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    
    plt.subplot(222)
    plt.plot(losses_per_episode, label="Loss", color='purple')
    plt.title("Policy Gradient Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return policy_net

if __name__ == "__main__":
    print("Starting policy gradient training...")
    trained_policy_net = train_policy_gradient(total_episodes=2000)
    print("Training finished, saving model to policy_net.pth")
    torch.save(trained_policy_net.state_dict(), "policy_net.pth")
    print("Done!")
