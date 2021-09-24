# This is the deep neural network used as the function approximator in Deep Reinformcement Learning
# ... for learning text-based exploration policies.

import pfrl             # A library for Deep Reinforcement Learning
import torch
import torch.nn  # PyTorch libraries


class q_function(torch.nn.Module):

    # Normally, observation size is equal to the number of state features.
    def __init__(self, observation_size, nb_actions):
        self.network_width = 1024
        self.observation_size = observation_size
        super().__init__()
        self.l1 = torch.nn.Linear(observation_size, self.network_width)
        self.l2 = torch.nn.Linear(self.network_width, self.network_width)
        self.l21 = torch.nn.Linear(self.network_width, self.network_width)
        self.l3 = torch.nn.Linear(self.network_width, nb_actions)

    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))
        h = torch.nn.functional.relu(self.l21(h))
        h = self.l3(h)
        return pfrl.action_value.DiscreteActionValue(h)
