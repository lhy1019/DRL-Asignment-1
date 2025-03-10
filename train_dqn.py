import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from custom_env import TrainingTaxiEnv  # Your environment
from simple_custom_taxi_env import SimpleTaxiEnv  # Simplified environment for training

# Define a simple feedforward neural network for approximating Q-values
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# Hyperparameters
num_episodes = 1000
batch_size = 32
gamma = 0.99
learning_rate = 0.001
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
replay_buffer_size = 5000

# Assuming your state is represented as a vector of fixed size (e.g., 10-20 features)
input_dim = 16  # Update this based on your state representation
output_dim = 6  # Number of actions: [South, North, East, West, Pickup, Dropoff]

# Initialize network, optimizer, and replay buffer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(input_dim, output_dim).to(device)
target_net = DQN(input_dim, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = deque(maxlen=replay_buffer_size)

# Function to convert state to tensor (you need to adjust state representation accordingly)
def state_to_tensor(state):
    # Example: flatten state tuple to a numpy array and then to a torch tensor
    return torch.tensor(np.array(state, dtype=np.float32), device=device)

# Main DQN training loop
env = SimpleTaxiEnv()  # Initialize your environment
for episode in range(num_episodes):
    # For diversity, reinitialize environment with different grid sizes and obstacles
    obs, _ = env.reset()
    
    # You would need a function to transform your observation to a fixed-size vector
    state = np.array(obs, dtype=np.float32)[:input_dim]  # Example transformation
    total_reward = 0
    done = False
    
    while not done:
        # ε-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = policy_net(state_to_tensor(state).unsqueeze(0))
            action = torch.argmax(q_values).item()
        
        next_obs, reward, done, truncated, _ = env.step(action)
        next_state = np.array(next_obs, dtype=np.float32)[:input_dim]
        
        # Store experience in replay buffer
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        
        # Training step
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states_tensor = state_to_tensor(np.vstack(states))
            actions_tensor = torch.tensor(actions, device=device).unsqueeze(1)
            rewards_tensor = torch.tensor(rewards, device=device).unsqueeze(1)
            next_states_tensor = state_to_tensor(np.vstack(next_states))
            dones_tensor = torch.tensor(dones, device=device, dtype=torch.float32).unsqueeze(1)
            
            # Compute current Q values and targets
            current_q = policy_net(states_tensor).gather(1, actions_tensor)
            with torch.no_grad():
                next_q = target_net(next_states_tensor).max(1)[0].unsqueeze(1)
            target_q = rewards_tensor + (gamma * next_q * (1 - dones_tensor))
            
            loss = nn.MSELoss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Update target network periodically
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode+1}/{num_episodes} | Total Reward: {total_reward} | Epsilon: {epsilon:.3f}")

print("Training complete.")
