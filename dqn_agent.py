# Remember to adjust your student ID in meta.xml
import numpy as np
import torch
import torch.nn as nn
import random

# Mapping of station indices to labels
stations = {0: 'R', 1: 'G', 2: 'Y', 3: 'B'}

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return self.fc3(x)  # Output Q-values for each action

input_dim = 19  # Our state vector is 23-dimensional
output_dim = 6  # Actions: 0: South, 1: North, 2: East, 3: West, 4: Pickup, 5: Dropoff
q_net = QNetwork(input_dim, output_dim).to(device)
q_net.load_state_dict(torch.load("q_net.pth", map_location=device))
q_net.eval()  # Set to evaluation mode

# --- Global Memory Variables ---
# These help the agent "remember" events within an episode.
pickup_flag = False
visited = [0, 0, 0, 0]
destination = None

# --- State Vector Conversion ---
def get_state_vector(obs, pickup=False, visited=[0,0,0,0], destination=None):
    """
    Converts the observation into a 23-dimensional state vector.
    The observation is expected to be a tuple with 16 elements:
      taxi_row, taxi_col,
      Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol,
      obstacle_north, obstacle_south, obstacle_east, obstacle_west,
      passenger_look, destination_look

    The state vector is constructed as:
      [pickup_flag (1 dim),
       visited (4 dims),
       destination one-hot (4 dims; zeros if destination is None),
       relative distances for R, G, Y, B (8 dims),
       obstacles (4 dims),
       passenger_look & destination_look (2 dims)]
    Total dims: 1+4+4+8+4+2 = 23.
    """
    taxi_row, taxi_col, Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol, \
         obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
         passenger_look, destination_look = obs

    # Relative distances: station position minus taxi position.
    drR, dcR = Rrow - taxi_row, Rcol - taxi_col
    drG, dcG = Grow - taxi_row, Gcol - taxi_col
    drY, dcY = Yrow - taxi_row, Ycol - taxi_col
    drB, dcB = Brow - taxi_row, Bcol - taxi_col

    pickup_val = 1.0 if pickup else 0.0
    visited_arr = np.array(visited, dtype=np.float32)

    # One-hot encoding for destination. If destination is None, use zeros.
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
        np.array([pickup_val], dtype=np.float32),
        # visited_arr,
        dest_vec,
        np.array([drR, dcR, drG, dcG, drY, dcY, drB, dcB], dtype=np.float32),
        obstacles,
        np.array([passenger_val, dest_look_val], dtype=np.float32)
    ))
    return state_vector

def locate_destination(stations_offset):
    """
    Given a list of offsets for the stations [(dr, dc), ...],
    returns the index of the closest station (using Manhattan distance).
    """
    distances = [abs(dr) + abs(dc) for dr, dc in stations_offset]
    return distances.index(min(distances))

# --- Action Selection Function ---
def get_action(obs):
    global pickup_flag, visited, destination
    # Update pickup_flag:
    # If the current observation shows that the passenger is no longer in view (e.g., passenger_look == 0),
    # assume a successful pickup.
    if not pickup_flag and obs[-2] == 0:
        pickup_flag = True

    # Update destination if destination_look is active.
    if obs[-1] == 1:
        taxi_row, taxi_col = obs[0], obs[1]
        offsets = [
            (obs[2] - taxi_row, obs[3] - taxi_col),  # R
            (obs[4] - taxi_row, obs[5] - taxi_col),  # G
            (obs[6] - taxi_row, obs[7] - taxi_col),  # Y
            (obs[8] - taxi_row, obs[9] - taxi_col)   # B
        ]
        dest_idx = locate_destination(offsets)
        destination = stations[dest_idx]

    # Update visited stations if taxi is at a station's coordinates.
    taxi_row, taxi_col = obs[0], obs[1]
    if taxi_row == obs[2] and taxi_col == obs[3]:
        visited[0] = 1
    if taxi_row == obs[4] and taxi_col == obs[5]:
        visited[1] = 1
    if taxi_row == obs[6] and taxi_col == obs[7]:
        visited[2] = 1
    if taxi_row == obs[8] and taxi_col == obs[9]:
        visited[3] = 1

    # Convert the observation and global memory into a state vector.
    state_vec = get_state_vector(obs, pickup_flag, visited, destination)
    state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Use the Q-network to choose the action with the highest Q-value.
    with torch.no_grad():
        q_values = q_net(state_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
    return action
