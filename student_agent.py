import numpy as np
import pickle
import random

# Load the pre-trained Q-table
with open("q_table.pkl", "rb") as f:
    Q_table = pickle.load(f)
directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
directions_counter = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
# Record important state information
agent_information = {
    "pickup": 0,
    "stations": [],
    "visited": set(),
    "visit_count": 0,
    "paseenger_pos": None,
    "prev_taxi_pos": None,
    "prev_direction": None,
    "target": None,
    "unvisited": set()
}

def reset_agent(obs):
    agent_information['pickup'] = 0
    agent_information["stations"] = [
        (obs[2], obs[3]),
        (obs[4], obs[5]),
        (obs[6], obs[7]),
        (obs[8], obs[9]),
    ]

    visited = set()
    visited.add((obs[0], obs[1]))
    agent_information["visited"] = visited
    agent_information["visit_count"] = 1
    agent_information["passenger_pos"] = None
    agent_information["prev_taxi_pos"] = (obs[0], obs[1])
    agent_information["prev_direction"] = None
    agent_information["target"] = np.random.randint(0, 4)
    agent_information['unvisited'] = set([0, 1, 2, 3])

    


def get_state(obs, visited=0, pickup=False, passenger_pos = None, target=None):
    # For this REINFORCE version we use a simplified state representation:
    # For example, we take only the obstacle flags.
    taxi_row, taxi_col, Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs  
                 
    dR = abs(taxi_row - Rrow) + abs(taxi_col - Rcol)
    dG = abs(taxi_row - Grow) + abs(taxi_col - Gcol)
    dY = abs(taxi_row - Yrow) + abs(taxi_col - Ycol)
    dB = abs(taxi_row - Brow) + abs(taxi_col - Bcol)
    
    passenger_dir = 5
    destination_dir = 5
    d_offset = (100, 100)
    
    if passenger_look:
        if passenger_pos:
            dP = abs(taxi_row - passenger_pos[0]) + abs(taxi_col - passenger_pos[1])
            if dP == 0:
                passenger_dir = 4
            else:
                passenger_offset = (passenger_pos[0] - taxi_row, passenger_pos[1] - taxi_col)
                try:
                    passenger_dir = directions.index(passenger_offset)
                except:
                    passenger_dir = 5
        else:   
            if min(dR, dG, dY, dB) == 0:
                passenger_dir = 4
            else:
                passenger_idx = np.argmin([dR, dG, dY, dB])
                if passenger_idx == 0:
                    passenger_offset = (Rrow - taxi_row, Rcol - taxi_col)
                elif passenger_idx == 1:
                    passenger_offset = (Grow - taxi_row, Gcol - taxi_col)
                elif passenger_idx == 2:
                    passenger_offset = (Yrow - taxi_row, Ycol - taxi_col)
                elif passenger_idx == 3:
                    passenger_offset = (Brow - taxi_row, Bcol - taxi_col)
                try:
                    passenger_dir = directions.index(passenger_offset)
                except:
                    passenger_dir = 5
                
                
    if destination_look:
        if min(dR, dG, dY, dB) == 0:
            destination_dir = 4
        else:
            destination_idx = np.argmin([dR, dG, dY, dB])
            if destination_idx == 0:
                destination_offset = (Rrow - taxi_row, Rcol - taxi_col)
            elif destination_idx == 1:
                destination_offset = (Grow - taxi_row, Gcol - taxi_col)
            elif destination_idx == 2:
                destination_offset = (Yrow - taxi_row, Ycol - taxi_col)
            elif destination_idx == 3:
                destination_offset = (Brow - taxi_row, Bcol - taxi_col)
            try:
                destination_dir = directions.index(destination_offset)
            except:
                destination_dir = 5
    if target is not None:
        if target == 0:
            d_offset = (Rrow - taxi_row, Rcol - taxi_col)
        elif target == 1:
            d_offset = (Grow - taxi_row, Gcol - taxi_col)
        elif target == 2:
            d_offset = (Yrow - taxi_row, Ycol - taxi_col)
        elif target == 3:
            d_offset = (Brow - taxi_row, Bcol - taxi_col)
    
        
          
    return (pickup,
            target, d_offset,
            obstacle_north, obstacle_south, obstacle_east, obstacle_west, 
            passenger_look, passenger_dir, 
            destination_look, destination_dir)



def get_action(obs, render=False):
    global agent_information, Q_table
    for i in range(4):
        if (obs[2*i+2], obs[2*i+3]) not in agent_information["stations"]:
            reset_agent(obs)
            # print("Resetting agent")
            break
        
    if agent_information["prev_taxi_pos"]:
        offset = (obs[0] - agent_information["prev_taxi_pos"][0], obs[1] - agent_information["prev_taxi_pos"][1])
        if abs(offset[0]) + abs(offset[1]) > 1:
            reset_agent(obs)

    if (obs[0], obs[1]) not in agent_information["visited"]:
        agent_information["visited"].add((obs[0], obs[1]))
        agent_information["visit_count"] += 1
        
    if agent_information['pickup']:
        agent_information['passenger_pos'] = (obs[0], obs[1])

    state = get_state(obs, pickup=agent_information["pickup"], 
                        visited=agent_information["visit_count"],
                        passenger_pos=agent_information["passenger_pos"],
                        target=agent_information['target'],
                      )

    if state not in Q_table:
        action = np.random.randint(0, 6)

    else:
        qvals = Q_table[state]
        sorted_actions = qvals.argsort()  # ascending order
        best_action = sorted_actions[-1]
        second_best_action = sorted_actions[-2]
        third_best_action = sorted_actions[-3]
        fourth_best_action = sorted_actions[-4]
        # With probability epsilon_second_best, pick second best action
        action = best_action
        # # if action in [0, 1, 2, 3]:
        # #     next_pos = (obs[0] + directions[action][0], obs[1] + directions[action][1])
        # #     if next_pos in agent_information["visited"]:
        # #         if np.random.uniform(0, 1) < 0.2:
        # #             action = second_best_action
        # prev_action = agent_information["prev_direction"]
        # # if prev_action and directions_counter.index(prev_action) == best_action:
        # #     if np.random.uniform(0, 1) < 0.5:
        # #         action = second_best_action
        if action not in [4, 5]:
            if np.random.uniform(0, 1) < 0.2:
                action = np.random.choice([best_action, second_best_action])
        if render:
            print(f"Action: {action}, State: {state}")
            print()
    d_target = abs(state[2][0]) + abs(state[2][1])
    if d_target == 0:
        agent_information['target'] = np.random.choice(list(agent_information['unvisited']))
    passenger_dir = state[-3]
    if not agent_information["pickup"] and passenger_dir == 4 and action == 4:
        agent_information["pickup"] = 1
        agent_information["passenger_pos"] = (obs[0], obs[1])
        
    if action == 5:
        agent_information["pickup"] = 0
    agent_information["prev_taxi_pos"] = (obs[0], obs[1])
    if action < 4:
        agent_information["prev_direction"] = directions[action] 
    return action
