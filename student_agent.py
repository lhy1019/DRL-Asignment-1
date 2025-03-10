import numpy as np
import pickle

# Load the pre-trained Q-table
with open("reinforce_policy.pkl", "rb") as f:
    Q_table = pickle.load(f)

# Record important state information
agent_information = {
    "prev_0_direction": (0, 0),
    "prev_1_direction": (0, 0),
    "pickup": 0,
    "stations": [],
}

def reset_agent(obs):
    agent_information['prev_0_direction'] = (0, 0)
    agent_information["prev_1_direction"] = (0, 0)
    agent_information['pickup'] = 0
    agent_information["stations"] = [
        (obs[2], obs[3]),
        (obs[4], obs[5]),
        (obs[6], obs[7]),
        (obs[8], obs[9]),
    ]
    

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)

directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
counter_directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

def get_state(obs, prev_0_direction=(0, 0), prev_1_direction=(0, 0), pickup=0):
    taxi_row,   taxi_col, \
    Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs  
    
    dR = abs(taxi_row - Rrow) + abs(taxi_col - Rcol)
    dG = abs(taxi_row - Grow) + abs(taxi_col - Gcol)
    dY = abs(taxi_row - Yrow) + abs(taxi_col - Ycol)
    dB = abs(taxi_row - Brow) + abs(taxi_col - Bcol)
    on_passenger = 0
    on_destination = 0
    
    if passenger_look and (min(dR, dG, dY, dB) == 0):
        on_passenger = 1
    if destination_look and (min(dR, dG, dY, dB) == 0):
        on_destination = 1
    return (pickup, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, on_passenger, destination_look, on_destination)
    return (obstacle_north, obstacle_south, obstacle_east, obstacle_west, prev_0_direction, prev_1_direction, passenger_look, on_passenger)


def get_action(obs):
    global agent_information, Q_table
    for i in range(4):
        if (obs[2*i+2], obs[2*i+3]) not in agent_information["stations"]:
            reset_agent(obs)
            # print("Resetting agent")
            break
    # state = get_state(obs, agent_information["prev_0_direction"], agent_information["prev_1_direction"])
    state = get_state(obs, pickup=agent_information["pickup"])

    if state not in Q_table:
        action = np.random.randint(0, 6)
        # print("Random action")
    else:
        logits = Q_table[state]
        probs = softmax(logits)
        action = np.random.choice(np.arange(6), p=probs)
        
    if action < 4:
        agent_information["prev_1_direction"] = agent_information["prev_0_direction"]
        agent_information["prev_0_direction"] = directions[action]
    if not agent_information["pickup"] and state[-3] and action == 4:
        agent_information['pickup'] = 1
        
    return action
