import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, activations=nn.ReLU()):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activations)
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class ValueCritic(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(ValueCritic, self).__init__()
        self.mlp = MLP(input_dim, hidden_dims)

    def forward(self, observations):
        critic = self.mlp(observations)
        return critic.squeeze(-1)

class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dims, activations=nn.ReLU()):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.concat_dim = input_dim + action_dim
        self.mlp = MLP(self.concat_dim, hidden_dims, activations=activations)

    def forward(self, observations, actions):
        inputs = torch.cat([observations, actions], dim=-1)
        critic = self.mlp(inputs)
        return critic.squeeze(-1)

class DoubleCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dims, activations=nn.ReLU()):
        super(DoubleCritic, self).__init__()
        self.critic1 = Critic(input_dim, action_dim, hidden_dims, activations=activations)
        self.critic2 = Critic(input_dim, action_dim, hidden_dims, activations=activations)

    def forward(self, observations, actions):
        critic1 = self.critic1(observations, actions)
        critic2 = self.critic2(observations, actions)
        return critic1, critic2
