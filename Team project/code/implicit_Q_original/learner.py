import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Optional, Sequence, Tuple, Dict, Any

import policy
import value_net
from actor import update as awr_update_actor
from common import Batch, InfoDict, Model

def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = {
        k: tau * critic.params[k] + (1 - tau) * target_critic.params[k]
        for k in critic.params
    }
    target_critic.update_params(new_target_params)
    return target_critic

def _update(
    rng: torch.Generator, actor: Model, critic: Model, value: Model,
    target_critic: Model, batch: Batch, discount: float, tau: float,
    expectile: float, temperature: float
) -> Tuple[torch.Generator, Model, Model, Model, Model, InfoDict]:

    new_value, value_info = value_net.update_v(target_critic, value, batch, expectile)
    key = torch.Generator().manual_seed(rng.seed() + 1)
    new_actor, actor_info = awr_update_actor(key, actor, target_critic, new_value, batch, temperature)
    
    new_critic, critic_info = value_net.update_q(critic, new_value, batch, discount)
    new_target_critic = target_update(new_critic, target_critic, tau)

    info = {**critic_info, **value_info, **actor_info}
    return key, new_actor, new_critic, new_value, new_target_critic, info

class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: torch.Tensor,
                 actions: torch.Tensor,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 temperature: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 opt_decay_schedule: str = "cosine"):

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.temperature = temperature

        torch.manual_seed(seed)
        rng = torch.Generator().manual_seed(seed)

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False)

        optimiser = optim.Adam(actor_def.parameters(), lr=actor_lr)
        
        if opt_decay_schedule == "cosine" and max_steps is not None:
            lr_lambda = lambda step: 0.5 * (1 + torch.cos(step * torch.pi / torch.tensor(max_steps)))
            self.scheduler = optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
        else:
            self.scheduler = None

        actor = Model(actor_def, optimiser, actor_def.parameters())

        critic_def = value_net.DoubleCritic(hidden_dims)
        critic_optim = optim.Adam(critic_def.parameters(), lr=critic_lr)
        critic = Model(critic_def, critic_optim, critic_def.parameters())


        value_def = value_net.ValueCritic(hidden_dims)
        value_optim = optim.Adam(value_def.parameters(), lr=value_lr)
        value = Model(value_def, value_optim, value_def.parameters())


        target_critic = Model(critic_def, critic_optim, critic_def.parameters())

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng

    def sample_actions(self,
                       observations: torch.Tensor,
                       temperature: float = 1.0) -> torch.Tensor:
        rng, actions = policy.sample_actions(self.rng, self.actor.apply_fn,
                                             self.actor.params, observations,
                                             temperature)
        self.rng = rng
        actions = torch.clamp(actions, -1, 1)
        return actions

    def update(self, batch: Batch) -> InfoDict:
        new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update(
            self.rng, self.actor, self.critic, self.value, self.target_critic,
            batch, self.discount, self.tau, self.expectile, self.temperature)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        if self.scheduler:
            self.scheduler.step()

        return info
