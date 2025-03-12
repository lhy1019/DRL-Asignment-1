import numpy as np
import pickle
import random

# Load the pre-trained Q-table
with open("reinforce_policy.pkl", "rb") as f:
    Q_table = pickle.load(f)

# Record important state information
agent_information = {
    "pickup": 0,
    "stations": [],
    "target": None,
    "unvisitied": set(["R", "G", "Y", "B"])
}

def reset_agent(obs):
    agent_information['pickup'] = 0
    agent_information["stations"] = [
        (obs[2], obs[3]),
        (obs[4], obs[5]),
        (obs[6], obs[7]),
        (obs[8], obs[9]),
    ]
    target = random.choice(["R", "G", "Y", "B"])
    agent_information["target"] = target
    agent_information["unvisitied"] = set(["R", "G", "Y", "B"])
    

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)

directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
counter_directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

def get_state(obs, pickup=False, target=None, arrived=False):
    # For this REINFORCE version we use a simplified state representation:
    # For example, we take only the obstacle flags.
    taxi_row, taxi_col, Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs  
    
    target_pos = (Rrow, Rcol) if target == "R" else \
                 (Grow, Gcol) if target == "G" else \
                 (Yrow, Ycol) if target == "Y" else \
                 (Brow, Bcol)
                 
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
        
    dr_target, dc_target = target_pos[0] - taxi_row, target_pos[1] - taxi_col  
    dT = abs(dr_target) + abs(dc_target) 
          
    return (pickup,
            target, dT,
            obstacle_north, obstacle_south, obstacle_east, obstacle_west, 
            passenger_look, on_passenger, 
            destination_look, on_destination)


def get_action(obs, render=False):
    global agent_information, Q_table
    for i in range(4):
        if (obs[2*i+2], obs[2*i+3]) not in agent_information["stations"]:
            reset_agent(obs)
            # print("Resetting agent")
            break
    
    # state = get_state(obs, agent_information["prev_0_direction"], agent_information["prev_1_direction"])
    state = get_state(obs, pickup=agent_information["pickup"], target=agent_information["target"])

    if state not in Q_table:
        action = np.random.randint(0, 6)
        # print("Random action")
    else:
        logits = Q_table[state]
        probs = softmax(logits)
        action = np.random.choice(np.arange(6), p=probs)
        if render:
            print(f"Action: {action}, State: {state}")
            print()
            
    if state[2] == 0:
        if not state[-1] and agent_information['target'] in agent_information["unvisitied"]:
            agent_information["unvisitied"].remove(agent_information["target"])
        if agent_information["unvisitied"]:
            agent_information["target"] = random.choice(list(agent_information["unvisitied"]))
        else:
            agent_information["target"] = random.choice(["R", "G", "Y", "B"])
        
    if not agent_information["pickup"] and state[-3] and action == 4:
        agent_information['pickup'] = 1
        
    return action
