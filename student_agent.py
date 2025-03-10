import numpy as np
import pickle

# Load the pre-trained Q-table
with open("q_table.pkl", "rb") as f:
    Q_table = pickle.load(f)

# Mapping station indices to letters
stations = {
    0: 'R',
    1: 'G',
    2: 'Y',
    3: 'B'
}

# Agent internal state, maintained between steps within an episode.
agent_internal_state = {
    "pickup": False,
    "visited": [0, 0, 0, 0],  # Order: R, G, Y, B
    "destination": None
}

# Global variables to help detect episode resets and track previous pickup observation.
prev_taxi_pos = None
prev_passenger_look = None
prev_action = None

def reset_agent():
    """Reset the agent's internal state and helper variables.
    Call this function when a new episode begins.
    """
    global agent_internal_state, prev_taxi_pos, prev_passenger_look, prev_action
    agent_internal_state = {
        "pickup": False,
        "visited": [0, 0, 0, 0],
        "destination": None
    }
    prev_taxi_pos = None
    prev_passenger_look = None
    prev_action = None

def locate_destination(stations_offset):
    """
    Given a tuple of (row, col) differences for stations R, G, Y, B,
    return the index of the station with the smallest Manhattan distance.
    """
    distances = [abs(dr) + abs(dc) for dr, dc in stations_offset]
    return distances.index(min(distances))

def get_state(obs):
    """
    Convert the raw observation into a state tuple.
    
    The observation format is:
      (taxi_row, taxi_col, Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol,
       obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
    
    The state tuple is defined as:
      (pickup, visited, destination,
       (drR, dcR), (drG, dcG), (drY, dcY), (drB, dcB),
       (obstacle_north, obstacle_south, obstacle_east, obstacle_west),
       passenger_look, destination_look)
       
    where:
      - pickup: whether the passenger has been picked up
      - visited: a tuple marking whether each station (R, G, Y, B) has been visited (1) or not (0)
      - destination: the target station letter (e.g. 'R') or None if not set yet
      - (drX, dcX): differences between taxi position and station X position
    """
    taxi_row, taxi_col, Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    
    drR, dcR = Rrow - taxi_row, Rcol - taxi_col
    drG, dcG = Grow - taxi_row, Gcol - taxi_col
    drY, dcY = Yrow - taxi_row, Ycol - taxi_col
    drB, dcB = Brow - taxi_row, Bcol - taxi_col
    
    pickup = agent_internal_state["pickup"]
    visited = tuple(agent_internal_state["visited"])
    destination = agent_internal_state["destination"]
    
    return (pickup, visited, destination,
            (drR, dcR), (drG, dcG), (drY, dcY), (drB, dcB),
            (obstacle_north, obstacle_south, obstacle_east, obstacle_west),
            passenger_look, destination_look)

def get_action(obs):
    """
    Given an observation from the environment, returns an action (0-5) based on the pre-trained Q-table.
    
    This function implements the revised pickup logic:
      - If the previous action was PICKUP (action 4) and in the previous observation the passenger was visible (passenger_look was 1)
        but now the passenger is no longer visible (passenger_look is 0), then the agent registers a successful pickup.
    
    Additionally, it checks for a new episode by detecting a sudden jump in taxi position.
    """
    global agent_internal_state, Q_table, prev_taxi_pos, prev_passenger_look, prev_action

    # Extract current taxi position from the observation (first two elements).
    current_taxi_pos = (obs[0], obs[1])
    if prev_taxi_pos is not None:
        # If the taxi's position jumped more than one cell (Manhattan distance > 1), assume a new episode.
        if abs(current_taxi_pos[0] - prev_taxi_pos[0]) + abs(current_taxi_pos[1] - prev_taxi_pos[1]) > 1:
            reset_agent()
    prev_taxi_pos = current_taxi_pos

    # Extract current passenger look from the observation (index -2, or obs[14]).
    current_passenger_look = obs[14]

    # Check pickup success:
    # If the previous action was PICKUP (4), and in the previous observation the passenger was visible,
    # but now the passenger is no longer visible, then set the pickup flag.
    if prev_action == 4 and prev_passenger_look is not None:
        if prev_passenger_look == 1 and current_passenger_look == 0:
            agent_internal_state["pickup"] = True

    # Update the previous passenger look for the next step.
    prev_passenger_look = current_passenger_look

    # Build the current state tuple from the observation and the internal state.
    state = get_state(obs)
    
    # Update destination if not yet set and destination_look is active.
    if agent_internal_state["destination"] is None and state[-1] == 1:
        stations_offset = state[3:7]  # Offsets for stations R, G, Y, B.
        dest_index = locate_destination(stations_offset)
        agent_internal_state["destination"] = stations[dest_index]
    
    # Mark a station as visited if taxi is at that station (offset equals (0, 0)).
    if state[3] == (0, 0) and agent_internal_state["visited"][0] == 0:
        agent_internal_state["visited"][0] = 1
    if state[4] == (0, 0) and agent_internal_state["visited"][1] == 0:
        agent_internal_state["visited"][1] = 1
    if state[5] == (0, 0) and agent_internal_state["visited"][2] == 0:
        agent_internal_state["visited"][2] = 1
    if state[6] == (0, 0) and agent_internal_state["visited"][3] == 0:
        agent_internal_state["visited"][3] = 1

    # Action selection based on the Q-table.
    # If the current state was not encountered during training, choose a random action.
    if state not in Q_table:
        action = np.random.randint(0, 6)
    else:
        action = np.argmax(Q_table[state])

    print(f"State: {state}, Action: {action}")
        
    # Store the current action as the previous action for the next call.
    prev_action = action

    return action
