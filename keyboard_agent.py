# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import curses

key_to_action = {
    'w': 1,      # Move North
    's': 0,    # Move South
    'a': 3,    # Move West
    'd': 2,   # Move East
    'q': 4,   # Pick up passenger
    'e': 5    # Drop off passenger
}

# with open("q_table.pkl", "rb") as f:
#     Q_table = pickle.load(f)
    
def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    try:
        action = key_to_action[input()]
    except KeyError:
        action = key_to_action[input("Invalid key. Please enter a valid key: ")]

    return action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

