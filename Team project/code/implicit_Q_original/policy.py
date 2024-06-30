import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from typing import Optional, Sequence, Tuple
import numpy as np

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class MLP(nn.Module):
    def __init__(self, hidden_dims: Sequence[int], activate_final: bool = True, dropout_rate: Optional[float] = None):
        super().__init__()
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(in_features=dim, out_features=dim))
            layers.append(nn.ReLU())
            if dropout_rate is not None:
                layers.append(nn.Dropout(p=dropout_rate))
        if not activate_final:
            layers = layers[:-1]  # Remove final activation if not required
        self.net = nn.Sequential(*layers)

    def forward(self, x, training: bool = False):
        if not training:
            self.net.eval()
        return self.net(x)


class NormalTanhPolicy(nn.Module):
    def __init__(self, hidden_dims: Sequence[int], action_dim: int, state_dependent_std: bool = True,
                 dropout_rate: Optional[float] = None, log_std_scale: float = 1.0, log_std_min: Optional[float] = None,
                 log_std_max: Optional[float] = None, tanh_squash_distribution: bool = True):
        super().__init__()
        self.mlp = MLP(hidden_dims, activate_final=True, dropout_rate=dropout_rate)
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
        else:
            self.log_stds = nn.Parameter(torch.zeros(action_dim))
        self.log_std_scale = log_std_scale
        self.log_std_min = log_std_min or LOG_STD_MIN
        self.log_std_max = log_std_max or LOG_STD_MAX
        self.tanh_squash_distribution = tanh_squash_distribution

    def forward(self, observations: torch.Tensor, temperature: float = 1.0, training: bool = False):
        outputs = self.mlp(observations, training=training)
        means = self.mean_layer(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_layer(outputs)
        else:
            log_stds = self.log_stds

        log_stds = torch.clamp(log_stds * self.log_std_scale, self.log_std_min, self.log_std_max)
        if not self.tanh_squash_distribution:
            means = torch.tanh(means)

        base_dist = Normal(loc=means, scale=torch.exp(log_stds) * temperature)
        if self.tanh_squash_distribution:
            return TransformedDistribution(base_dist, TanhTransform())
        else:
            return base_dist


def sample_actions(rng: torch.Generator, actor: NormalTanhPolicy, observations: np.ndarray, temperature: float = 1.0) -> Tuple[torch.Generator, torch.Tensor]:
    observations = torch.tensor(observations, dtype=torch.float32)
    dist = actor(observations, temperature)
    rng, key = torch.Generator(), torch.Generator().manual_seed(rng.seed() + 1)
    actions = dist.sample(sample_shape=observations.shape[:-1], generator=key)
    return rng, actions

