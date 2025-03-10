import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from custom_env import TrainingTaxiEnv  # Your environment

# --- Global Definitions ---
stations = {0: 'R', 1: 'G', 2: 'Y', 3: 'B'}

# Global memory variables to "remember" events during an episode.
# These are reset at the start of each episode.
pickup_flag = False
visited = [0, 0, 0, 0]
destination = None

# --- State Vector Conversion ---
def get_state_vector(obs, pickup=False, visited=[0,0,0,0], destination=None):
    """
    Converts the observation into a 23-dimensional vector.
    Observation (obs) is assumed to be a tuple with 16 elements:
      taxi_row, taxi_col,
      Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol,
      obstacle_north, obstacle_south, obstacle_east, obstacle_west,
      passenger_look, destination_look
    We construct the state vector as:
      [pickup (1),
       visited (4),
       destination (one-hot 4),  (if destination is None, use zeros)
       relative distances for R, G, Y, B (2 dims each = 8),
       obstacles (4),
       passenger_look (1),
       destination_look (1)]
    Total dims: 1+4+4+8+4+1+1 = 23.
    """
    taxi_row, taxi_col, Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol, \
        obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
        passenger_look, destination_look = obs

    # Relative distances: station coordinate minus taxi coordinate.
    drR, dcR = Rrow - taxi_row, Rcol - taxi_col
    drG, dcG = Grow - taxi_row, Gcol - taxi_col
    drY, dcY = Yrow - taxi_row, Ycol - taxi_col
    drB, dcB = Brow - taxi_row, Bcol - taxi_col

    pickup_val = 1.0 if pickup else 0.0
    visited_arr = np.array(visited, dtype=np.float32)

    # One-hot encode destination: if destination is not None, set the corresponding index.
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
        np.array([pickup_val], dtype=np.float32),   # 1 dim
        # visited_arr,                                  # 4 dims
        dest_vec,                                     # 4 dims
        np.array([drR, dcR, drG, dcG, drY, dcY, drB, dcB], dtype=np.float32),  # 8 dims
        obstacles,                                    # 4 dims
        np.array([passenger_val, dest_look_val], dtype=np.float32)   # 2 dims
    ))
    return state_vector  # shape (23,)

# --- Q-Network Definition ---
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # outputs Q-values for each action

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 19
output_dim = 6  # actions: 0: South, 1: North, 2: East, 3: West, 4: Pickup, 5: Dropoff
q_net = QNetwork(input_dim, output_dim).to(device)
optimizer = optim.Adam(q_net.parameters(), lr=0.001)
criterion = nn.MSELoss()

def locate_destination(stations_offset):
    """
    Given a list of offsets for stations [(dr, dc), ...] for R, G, Y, B,
    returns the index of the closest station (using Manhattan distance).
    """
    distances = [abs(dr) + abs(dc) for dr, dc in stations_offset]
    return distances.index(min(distances))

# --- DQN Training Function ---
def train_dqn(
    total_episodes=2000,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.9995,
    epsilon_min=0.01
):
    env = TrainingTaxiEnv(n=5, max_fuel=5000, obstacle_prob=0)
    rewards_per_episode = []
    avg_rewards_100 = []
    epsilons = []
    td_errors_per_episode = []
    rewards_buffer = deque(maxlen=100)
    
    global pickup_flag, visited, destination
    
    for episode in range(total_episodes):
        # Reset global memory variables for new episode
        pickup_flag = False
        visited = [0, 0, 0, 0]
        destination = None
        
        # Create a new environment instance
        obs, _ = env.reset(seed=1)
        state_vec = get_state_vector(obs, pickup_flag, visited, destination)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
        
        done = False
        episode_reward = 0.0
        episode_td_error_sum = 0.0
        episode_steps = 0
        prev_passenger_look = obs[-2]
        
        while not done:
            # Epsilon-greedy action selection from Q-network
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(state_tensor)
                    action = int(torch.argmax(q_values, dim=1).item())
            
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # --- Update Global Memory ---
            # If action 4 (Pickup) is executed and passenger look changes, update pickup_flag.
            current_passenger_look = next_obs[-2]
            if action == 4 and (current_passenger_look != prev_passenger_look) and not pickup_flag:
                pickup_flag = True
                reward += 100  # bonus reward for successful pickup
            
            # Update destination when destination_look is active.
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
            
            # Update visited stations if taxi reaches station coordinates.
            taxi_row, taxi_col = next_obs[0], next_obs[1]
            if taxi_row == next_obs[2] and taxi_col == next_obs[3]:
                visited[0] = 1
            if taxi_row == next_obs[4] and taxi_col == next_obs[5]:
                visited[1] = 1
            if taxi_row == next_obs[6] and taxi_col == next_obs[7]:
                visited[2] = 1
            if taxi_row == next_obs[8] and taxi_col == next_obs[9]:
                visited[3] = 1
            
            # Optionally, add any additional reward shaping here
            
            reward = reward  # (you can subtract a step penalty if desired)
            
            # Convert next observation to state vector and tensor.
            next_state_vec = get_state_vector(next_obs, pickup_flag, visited, destination)
            next_state_tensor = torch.tensor(next_state_vec, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Compute target Q-value
            with torch.no_grad():
                q_next = q_net(next_state_tensor)
                max_q_next = torch.max(q_next).item()
            target = reward + (gamma * max_q_next if not done else 0.0)
            
            # Compute current Q-value for the taken action.
            q_val = q_net(state_tensor)[0, action]
            loss = criterion(q_val, torch.tensor(target, dtype=torch.float32, device=device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            td_error = abs(target - q_val.item())
            episode_td_error_sum += td_error
            episode_reward += reward
            episode_steps += 1
            
            # Prepare for next step
            state_tensor = next_state_tensor
            prev_passenger_look = current_passenger_look
        
        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        rewards_per_episode.append(episode_reward)
        rewards_buffer.append(episode_reward)
        avg_rewards_100.append(np.mean(rewards_buffer))
        epsilons.append(epsilon)
        td_errors_per_episode.append(episode_td_error_sum / max(1, episode_steps))
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{total_episodes} | Avg Reward: {avg_rewards_100[-1]:.2f} | Epsilon: {epsilon:.3f} | Steps: {episode_steps}")
            # Optionally, render the final state of the episode.
            print("Final state obs:", next_obs)
            env.render()
            print()
            
    # Plot metrics
    plt.figure(figsize=(14, 6))
    plt.subplot(221)
    plt.plot(rewards_per_episode, label="Episode Reward")
    plt.plot(avg_rewards_100, label="Avg Reward (100 eps)", color='red')
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    
    plt.subplot(222)
    plt.plot(epsilons, label="Epsilon", color='green')
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.legend()

    plt.subplot(223)
    plt.plot(td_errors_per_episode, label="TD Error", color='purple')
    plt.title("TD Error (per Episode)")
    plt.xlabel("Episode")
    plt.ylabel("TD Error")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return q_net

if __name__ == "__main__":
    print("Starting DQN training...")
    trained_q_net = train_dqn(total_episodes=1000)
    print("Training finished, saving model to q_net.pth")
    torch.save(trained_q_net.state_dict(), "q_net.pth")
    print("Done!")
