import numpy as np
import pickle
import random

# Load the pre-trained Q-table
with open("q_table.pkl", "rb") as f:
    Q_table = pickle.load(f)

stations = {
    0: 'R',
    1: 'G',
    2: 'Y',
    3: 'B'
}

# Global variables to track episode state
pickup = False
prev_passenger_look = False
prev_action = None
visited = [0, 0, 0, 0]
destination = None

def locate_destination(stations_offset):
    """
    Given a list of four station offsets (differences from the taxi's position),
    return the index of the station with the smallest Manhattan distance.
    """
    distances = [abs(dr) + abs(dc) for dr, dc in stations_offset]
    return distances.index(min(distances))

def get_state(obs, pickup=False, visited=[0, 0, 0, 0], destination=None):
    """
    Transform the raw observation into a custom state tuple.
    The state consists of:
      - pickup flag (whether the passenger has been picked up)
      - visited stations (tuple of 4 ints)
      - destination (the station label, if determined)
      - relative offsets to the four stations (R, G, Y, B)
      - obstacle info (north, south, east, west)
      - passenger look flag
      - destination look flag
    """
    taxi_row, taxi_col, \
    Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs  
    
    # Calculate relative distances (offsets) for each station
    drR, dcR = Rrow - taxi_row, Rcol - taxi_col
    drG, dcG = Grow - taxi_row, Gcol - taxi_col
    drY, dcY = Yrow - taxi_row, Ycol - taxi_col
    drB, dcB = Brow - taxi_row, Bcol - taxi_col

    # Convert visited list to tuple for consistent dictionary keys
    visited = tuple(visited)
    return (pickup, visited, destination, 
            (drR, dcR), (drG, dcG), (drY, dcY), (drB, dcB),
            (obstacle_north, obstacle_south, obstacle_east, obstacle_west),
            passenger_look, destination_look)

def get_action(obs):
    """
    Select an action based on the current observation.
    If the state is unseen in the Q-table, a random action is chosen.
    """
    global pickup, prev_passenger_look, prev_action, visited, destination
    
    # Check current visibility of the passenger
    now_passenger_look = obs[-2]
    
    # If the passenger was visible in the previous step and a PICKUP (4) was attempted,
    # but is no longer visible, assume the passenger has been picked up.
    if (not now_passenger_look) and prev_passenger_look and prev_action == 4:
        pickup = True
        
    # Build the state tuple from the observation and current flags
    state = get_state(obs, pickup, visited=visited, destination=destination)
    
    # If the state is not in the Q-table, choose a random action from the full action space (0-5)
    if state not in Q_table:
        action = random.choice(list(range(6)))
        print("Unseen state encountered, taking random action:", action)
    else:
        action = int(np.argmax(Q_table[state]))
    
    # If the destination is sensed nearby, update the destination using all four station offsets
    if state[-1]:
        destination = locate_destination(state[3:7])  # use offsets for R, G, Y, and B
        destination = stations[destination]
    
    # Update visited flags if taxi is at a station (i.e. offset is (0,0))
    if state[3] == (0, 0):
        visited[0] = 1  
    if state[4] == (0, 0):
        visited[1] = 1
    if state[5] == (0, 0):
        visited[2] = 1
    if state[6] == (0, 0):
        visited[3] = 1
    
    # (Optional) Print the updated state and chosen action for debugging
    state = get_state(obs, pickup, visited=visited, destination=destination)
    print("State:", state, "Action:", action)
    
    prev_passenger_look = now_passenger_look
    prev_action = action
    return action
