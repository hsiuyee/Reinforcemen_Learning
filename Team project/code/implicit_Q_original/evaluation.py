from typing import Dict
import torch
import gym
import numpy as np

def evaluate(agent: torch.nn.Module, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        observation = env.reset()
        done = False

        while not done:
            # Convert observation to tensor
            observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                action = agent.sample_actions(observation_tensor, temperature=0.0).squeeze(0).numpy()

            observation, _, done, info = env.step(action)

        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
