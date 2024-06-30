import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

# Assuming Batch, InfoDict, and Model classes are already defined
# from the previous provided PyTorch code.

# Loss function
def loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

def update_v(critic: Model, value: Model, batch: Batch, expectile: float, value_optimizer: torch.optim.Optimizer) -> Tuple[Model, InfoDict]:
    actions = batch.actions
    q1 = critic(batch.observations, actions)
    q2 = critic(batch.observations, actions)  # Assuming the critic has two heads
    q = torch.minimum(q1, q2)

    def value_loss_fn() -> Tuple[torch.Tensor, InfoDict]:
        v = value(batch.observations)
        value_loss = loss(q - v, expectile).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
        }

    value_optimizer.zero_grad()
    value_loss, info = value_loss_fn()
    value_loss.backward()
    value_optimizer.step()

    return value, info

def update_q(critic: Model, target_value: Model, batch: Batch, discount: float, critic_optimizer: torch.optim.Optimizer) -> Tuple[Model, InfoDict]:
    next_v = target_value(batch.next_observations)

    target_q = batch.rewards + discount * batch.masks * next_v

    def critic_loss_fn() -> Tuple[torch.Tensor, InfoDict]:
        q1 = critic(batch.observations, batch.actions)
        q2 = critic(batch.observations, batch.actions)  # Assuming the critic has two heads
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    critic_optimizer.zero_grad()
    critic_loss, info = critic_loss_fn()
    critic_loss.backward()
    critic_optimizer.step()

    return critic, info
