import collections
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations']
)

def default_init(scale: Optional[float] = torch.sqrt(torch.tensor(2.0))):
    def init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return init

PRNGKey = Any
Params = Dict[str, Any]
Shape = Sequence[int]
Dtype = Any
InfoDict = Dict[str, float]

class MLP(nn.Module):
    def __init__(self, hidden_dims: Sequence[int], 
                 activations: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 activate_final: bool = False, dropout_rate: Optional[float] = None):
        super(MLP, self).__init__()
        layers = []
        for i, size in enumerate(hidden_dims):
            layers.append(nn.Linear(size, size))
            if i + 1 < len(hidden_dims) or activate_final:
                layers.append(activations)
                if dropout_rate is not None:
                    layers.append(nn.Dropout(dropout_rate))
        self.net = nn.Sequential(*layers)
        self.net.apply(default_init())

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        if training and hasattr(self, 'dropout_rate'):
            self.net.train()
        else:
            self.net.eval()
        return self.net(x)

class Model:
    def __init__(self, step: int, apply_fn: nn.Module, params: Params, 
                 optimizer: Optional[optim.Optimizer] = None):
        self.step = step
        self.apply_fn = apply_fn
        self.params = params
        self.optimizer = optimizer

    @classmethod
    def create(cls, model_def: nn.Module, inputs: Sequence[torch.Tensor], 
               optimizer_class: Optional[optim.Optimizer] = None) -> 'Model':
        model = model_def(*inputs)
        params = model.state_dict()

        if optimizer_class is not None:
            optimizer = optimizer_class(model.parameters())
        else:
            optimizer = None

        return cls(step=1, apply_fn=model_def, params=params, optimizer=optimizer)

    def __call__(self, *args, **kwargs):
        return self.apply_fn(*args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, 'Model']:
        self.optimizer.zero_grad()
        loss, info = loss_fn()
        loss.backward()
        self.optimizer.step()

        return self, info

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.apply_fn.state_dict(), save_path)

    def load(self, load_path: str) -> 'Model':
        self.apply_fn.load_state_dict(torch.load(load_path))
        return self
