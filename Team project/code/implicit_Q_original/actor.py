from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define the Batch and Model data structures
class Batch:
    def __init__(self, observations: torch.Tensor, actions: torch.Tensor):
        self.observations = observations
        self.actions = actions

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define the architecture here, e.g., a simple MLP for the actor and critic
        self.fc1 = nn.Linear(24, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)  # Adjust the output size as needed

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

def update(key: torch.Generator, actor: Model, critic: Model, value: Model,
           batch: Batch, temperature: float, actor_optimizer: optim.Optimizer) -> Tuple[Model, Dict[str, torch.Tensor]]:
    v = value(batch.observations)

    q1 = critic(batch.observations, batch.actions)
    q2 = critic(batch.observations, batch.actions)  # Assuming the critic has two heads
    q = torch.minimum(q1, q2)
    exp_a = torch.exp((q - v) * temperature)
    exp_a = torch.minimum(exp_a, torch.tensor(100.0))

    def actor_loss_fn() -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        dist = actor(batch.observations)  # Assuming the actor outputs a distribution
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss, {'actor_loss': actor_loss, 'adv': q - v}

    # Zero the gradients
    actor_optimizer.zero_grad()

    # Compute the loss
    actor_loss, info = actor_loss_fn()

    # Backpropagate the loss
    actor_loss.backward()

    # Update the parameters
    actor_optimizer.step()

    return actor, info