import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import pickle
from custom_env import TrainingTaxiEnv  # Your environment
import random


stations = {
    0: 'R',
    1: 'G',
    2: 'Y',
    3: 'B'
}
def get_state(obs, pickup=False, visited=[0,0,0,0], destination = None):
    taxi_row,   taxi_col, \
    Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs  
    
    drR, dcR = Rrow - taxi_row, Rcol - taxi_col
    drG, dcG = Grow - taxi_row, Gcol - taxi_col
    drY, dcY = Yrow - taxi_row, Ycol - taxi_col
    drB, dcB = Brow - taxi_row, Bcol - taxi_col

    pickup = pickup
    visited = tuple(visited)
    destination = destination
    return (pickup, visited, destination, (drR, dcR), (drG, dcG), (drY, dcY), (drB, dcB), (obstacle_north, obstacle_south, obstacle_east, obstacle_west), passenger_look, destination_look)

def locate_destination(stations_offset):
    distances = [abs(dr) + abs(dc) for dr, dc in stations_offset]
    return distances.index(min(distances))

def train_q_learning(
    total_episodes=50000,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.9995,
    epsilon_min=0.01,
):
    """
    Train a tabular Q-learning agent on the TrainingTaxiEnv environment.
    Logs episode rewards, average rewards, epsilon, and a simple TD error metric.
    Returns:
      - Q (dict): Dictionary mapping state -> np.array of action values.
    """

    # --- 1) Create Environment ---
    env = TrainingTaxiEnv(n=5, max_fuel=50000, obstacle_prob=0.1)
    
    # The Q-table is a dictionary: Q[state] = np.array of action-values, shape = (num_actions,)
    Q = defaultdict(lambda: np.zeros(env.action_space.n, dtype=np.float32))

    # --- 2) Metrics to Track ---
    rewards_per_episode = []
    avg_rewards_100 = []
    epsilons = []
    td_errors_per_episode = []  # We'll track average TD error as a "loss" measure
    pickup_successes = []  # Track successful pickups
    goal_successes = []  # Track goal completions
    step_counts = []  # Track steps per episode

    # Keep a rolling buffer for average reward
    rewards_buffer = deque(maxlen=100)

    # --- 3) Training Loop ---
    for episode in range(total_episodes):
        env = TrainingTaxiEnv(n=random.randint(5, 10), max_fuel=5000, obstacle_prob=random.uniform(0.0, 0.3))
        obs, _ = env.reset()    
        state = get_state(obs, pickup=False)

        done = False
        prev_passenger_look = state[-2]
        pickup = state[0]
        
        episode_reward = 0.0
        episode_td_error_sum = 0.0
        episode_steps = 0
        reached_goal = False
        
        visited = [0, 0, 0, 0]
        R_bonus = False
        G_bonus = False
        Y_bonus = False
        B_bonus = False
        
        destination = None

        while not done:
            if state not in Q:
                Q[state] = np.zeros(env.action_space.n, dtype=np.float32)
                
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = get_state(next_obs, pickup=pickup, visited=visited, destination=destination)
            
            shape_reward = 0
            if action == 4 and (next_state[-2] != prev_passenger_look) and not pickup:
                pickup = True
                shape_reward += 20
                
            if state[-1]:
                destination = locate_destination(state[3:7])
                destination = stations[destination]
                
            if state[3] == (0, 0) and not R_bonus:
                R_bonus = True
                visited[0] = 1
                # shape_reward += 5
            if state[4] == (0, 0) and not G_bonus:
                G_bonus = True
                visited[1] = 1
                # shape_reward += 5
            if state[5] == (0, 0) and not Y_bonus:
                Y_bonus = True
                visited[2] = 1
                # shape_reward += 5
            if state[6] == (0, 0) and not B_bonus:
                B_bonus = True
                visited[3] = 1
                # shape_reward += 5
                
            if done and episode_steps < 200:
                shape_reward += 50
                reached_goal = True
        
            # reward -= 0.1
            reward += shape_reward
            # Q-learning update
            q_vals = Q[state]
            q_vals_next = Q[next_state]

            td_target = reward + (gamma * np.max(q_vals_next) if not done else 0.0)
            td_error = td_target - q_vals[action]  # difference
            q_vals[action] += alpha * td_error

            # Accumulate metrics
            episode_reward += reward
            episode_td_error_sum += abs(td_error)
            episode_steps += 1
            

            # Move on
            state = (next_state)
            prev_passenger_look = state[-2]
            
                

        # Epsilon decay
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Store metrics
        rewards_per_episode.append(episode_reward)
        rewards_buffer.append(episode_reward)
        avg_rewards_100.append(np.mean(rewards_buffer))
        epsilons.append(epsilon)
        td_errors_per_episode.append(episode_td_error_sum / max(1, episode_steps))
        goal_successes.append(int(reached_goal))
        pickup_successes.append(int(pickup))
        step_counts.append(episode_steps)
        

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            # print(state)
            print(f"Episode {episode+1}/{total_episodes} | "
                  f"Avg Reward: {avg_rewards_100[-1]:.2f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Success Rate: {np.sum(goal_successes[-100:])}% | "
                  f"Pickup Rate: {np.sum(pickup_successes[-100:])}% | "
                  f"Avg Steps: {np.mean(step_counts[-100:]):.1f} | " 
                  f"Max Steps: {np.max(step_counts[-100:])} | "
                  f"Min Steps: {np.min(step_counts[-100:])}") 
            print(f"Final state: {state}")
            env.render()
            print()

    # --- 4) Plotting the Training Metrics ---
    plot_metrics(rewards_per_episode, avg_rewards_100, epsilons, td_errors_per_episode)

    # --- 5) Return or Save the Learned Q-table ---
    return dict(Q)  # Convert defaultdict to dict for safer usage

def plot_metrics(rewards, avg_rewards_100, epsilons, td_errors):
    """ Plot reward curves, epsilon decay, and TD-error trends. """
    plt.figure(figsize=(14, 6))
    plt.subplot(221)
    plt.plot(rewards, label="Episode Reward")   
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
    plt.plot(td_errors, label="TD Error", color='purple')
    plt.title("TD Error (per Episode)")
    plt.xlabel("Episode")
    plt.ylabel("TD Error")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

  
if __name__ == "__main__":
    print("Starting training...")
    q_table = train_q_learning()  # You can adjust episodes

    # Save the model's state_dict (best practice in PyTorch)

    print("Training finished, model saved to q_table.pkl")

    # If you prefer a single-file approach, you can also do:
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

    print("Done!")
