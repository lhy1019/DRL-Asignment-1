import numpy as np
import pickle

# Load the pre-trained Q-table
with open("q_table.pkl", "rb") as f:
    Q_table = pickle.load(f)


# Agent internal state, maintained between steps within an episode.
# agent_internal_state = {
#     "visited": set()
# }

# prev_taxi_pos = None
# stations_pos = []

# def reset_agent(obs):
#     global prev_taxi_pos, stations_pos
#     agent_internal_state['visited'] = set()
#     prev_taxi_pos = None
#     stations_pos = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]


def get_state(obs):
    taxi_row,   taxi_col, \
    Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs  
    
    return (obstacle_north, obstacle_south, obstacle_east, obstacle_west)


def get_action(obs):
    """
    Given an observation from the environment, returns an action (0-5) based on the pre-trained Q-table.
    
    This function implements the revised pickup logic:
      - If the previous action was PICKUP (action 4) and in the previous observation the passenger was visible (passenger_look was 1)
        but now the passenger is no longer visible (passenger_look is 0), then the agent registers a successful pickup.
    
    Additionally, it checks for a new episode by detecting a sudden jump in taxi position.
    """
    # global agent_internal_state, Q_table, prev_taxi_pos, stations_pos

    # # Extract current taxi position from the observation (first two elements).
    # current_taxi_pos = (obs[0], obs[1])
    # if prev_taxi_pos is not None:
    #     # If the taxi's position jumped more than one cell (Manhattan distance > 1), assume a new episode.
    #     if abs(current_taxi_pos[0] - prev_taxi_pos[0]) + abs(current_taxi_pos[1] - prev_taxi_pos[1]) > 1:
    #         reset_agent(obs)
            
    # if (obs[2], obs[3]) not in stations_pos:
    #     reset_agent(obs)
    # if (obs[4], obs[5]) not in stations_pos:
    #     reset_agent(obs)
    # if (obs[6], obs[7]) not in stations_pos:
    #     reset_agent(obs)
    # if (obs[8], obs[9]) not in stations_pos:
    #     reset_agent(obs)
 
    # prev_taxi_pos = current_taxi_pos

    # Build the current state tuple from the observation and the internal state.
    state = get_state(obs)

    # Action selection based on the Q-table.
    # If the current state was not encountered during training, choose a random action.
    if state not in Q_table:
        action = np.random.randint(0, 6)
        print("Random action")
    else:
        action = np.argmax(Q_table[state])
        
    return action
