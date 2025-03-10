import time
import random
import numpy as np
from custom_env import TrainingTaxiEnv, EvaluationTaxiEnv
from keyboard_agent import get_action



def test_env(env_class, num_episodes=5, render=True):
    """
    Test the given environment class with a random policy.
    """
    print(f"\nTesting {env_class.__name__}...")
    env = env_class(n=5, max_fuel=5000, obstacle_prob=0.1)

    for episode in range(num_episodes):
        print(f"\nEpisode {episode+1}")
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        env.render()
        while not done:
            action = get_action(obs)  # Use a random action
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            step_count += 1
            print(f'\nobs={obs}, reward={reward}, done={done}')
            if render:
                env.render()
                time.sleep(0.3)  # Slow down for visibility


        print(f"Episode {episode+1} finished in {step_count} steps with total reward {total_reward}\n")

    env.close()


if __name__ == "__main__":
    test_env(TrainingTaxiEnv, num_episodes=3, render=True)
    test_env(EvaluationTaxiEnv, num_episodes=3, render=True)
