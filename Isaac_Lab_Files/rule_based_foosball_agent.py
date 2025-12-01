import gymnasium as gym
import torch
import numpy as np


class RuleBasedFoosballAgent:
    def __init__(self):
        self.action_dim = 8

    def compute_action(self, obs):
        # Convert to numpy for easier rule coding (optional)
        if torch.is_tensor(obs):
            obs = obs.cpu().numpy()

        # TODO: get the meaningful information out of obs

        # TODO: calculate and set action 
        action = np.zeros(self.action_dim, dtype=np.float32)

        # makes action in range of -1 to 1 only 
        action = np.clip(action, -1.0, 1.0)

        return action
