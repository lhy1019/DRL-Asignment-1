import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import gym
import pickle
from custom_env import TrainingTaxiEnv  # or wherever your environment is

# -----------------------------------------
# 1) Define the Q-Network
# -----------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # A simple 2-layer MLP
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)  # Outputs Q-values for each action


# -----------------------------------------
# 2) Utility: convert state -> torch.Tensor
# -----------------------------------------
def state_to_tensor(state, device):
    """
    Convert state (tuple) to a FloatTensor on the given device.
    We'll flatten the tuple into a 1D tensor.
    """
    # If your state is a tuple, just convert to np.array then to torch tensor
    state_array = np.array(state, dtype=np.float32)
    return torch.tensor(state_array, dtype=torch.float32, device=device).unsqueeze(0)


# -----------------------------------------
# 3) Training Function
# -----------------------------------------
def train_dqn(
    total_episodes=5000,
    alpha=1e-3,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.9995,
    epsilon_min=0.01
):
    # Create environment
    env = TrainingTaxiEnv(n=5, max_fuel=5000, obstacle_prob=0)
    obs_example, _ = env.reset()  # A single example observation
    state_dim = len(obs_example)  # Flattened dimension of the state (your tuple length)
    action_dim = env.action_space.n

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create Q-Network
    q_network = QNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=alpha)
    loss_fn = nn.MSELoss()
    
    rewards_per_episode = []
    avg_rewards = []
    epsilons = []
    losses = []
    episode_lengths = []
    q_value_history = []
    
    reward_buffer = deque(maxlen=100)

    # Training Loop
    for episode in range(total_episodes):
        obs, _ = env.reset()
        state = state_to_tensor(obs, device)
        done = False
        episode_reward = 0
        steps = 0
        episode_loss = 0
        q_values_episode = []
        

        while not done or steps < 3000:
            # Epsilon-greedy for action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_network(state)  # shape [1, action_dim]
                action = torch.argmax(q_values, dim=1).item()
                q_values_episode.append(q_values.max().item())

            # Step in the environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Convert to tensors
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
            next_state = state_to_tensor(next_obs, device)

            # Compute TD target
            with torch.no_grad():
                if not done:
                    next_q_values = q_network(next_state)  # shape [1, action_dim]
                    max_next_q = torch.max(next_q_values, dim=1)[0]
                    td_target = reward_tensor + gamma * max_next_q
                else:
                    td_target = reward_tensor

            # Current Q(s,a)
            current_q_values = q_network(state)
            q_value = current_q_values[0, action]  # Q(s, a)

            # Compute loss
            loss = loss_fn(q_value.unsqueeze(0), td_target)

            # Gradient Descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            episode_reward += reward
            episode_loss += loss.item()
            steps += 1

            # Move on
            state = next_state

        # Epsilon decay
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
        # Store metrics
        rewards_per_episode.append(episode_reward)
        reward_buffer.append(episode_reward)
        avg_rewards.append(np.mean(reward_buffer))  # Moving average of last 100 episodes
        epsilons.append(epsilon)
        losses.append(episode_loss / steps)  # Avg loss per step
        episode_lengths.append(steps)
        q_value_history.append(np.mean(q_values_episode) if q_values_episode else 0)

        # Print progress
        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1}/{total_episodes}, Avg Reward: {avg_rewards[-1]:.2f}, Epsilon: {epsilon:.3f}")

    # Return the trained model
    return q_network


# -----------------------------------------
# 4) Main Entry: train & save model
# -----------------------------------------
if __name__ == "__main__":
    print("Starting DQN training...")
    q_model = train_dqn(total_episodes=300)  # You can adjust episodes

    # Save the model's state_dict (best practice in PyTorch)
    torch.save(q_model.state_dict(), "q_model.pth")
    print("Training finished, model saved to q_model.pth")

    # If you prefer a single-file approach, you can also do:
    # with open("q_model.pkl", "wb") as f:
    #     pickle.dump(q_model, f)

    print("Done!")
