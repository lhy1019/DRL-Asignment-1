import time
from custom_env import TrainingTaxiEnv, EvaluationTaxiEnv
import importlib.util
import random

def run_agent(agent_file, env_config, render=False, episodes=1):
    # Dynamically load the student_agent module from the specified file
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    # Create the environment using the provided configuration
    env = TrainingTaxiEnv(**env_config)
    total_rewards = []

    for episode in range(episodes):
        # Optionally vary the seed for each episode for more diverse testing
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            if render:
                env.render()
                time.sleep(0.3)
            # Use the dynamically loaded agent's get_action function
            action = student_agent.get_action(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            step_count += 1

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Steps = {step_count}, Reward = {total_reward}")

    avg_reward = sum(total_rewards) / episodes
    print(f"Average Reward over {episodes} episodes: {avg_reward}")
    return avg_reward

if __name__ == "__main__":
    env_config = {
        "n": random.randint(5, 10),
        "max_fuel": 5000,
        "obstacle_prob": random.uniform(0.0, 0.3)
    }
    
    agent_score = run_agent("student_agent.py", env_config, render=True)
    print(f"Final Score: {agent_score}")
