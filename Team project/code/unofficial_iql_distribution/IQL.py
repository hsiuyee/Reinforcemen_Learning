import copy
import torch

import os
from actor import Actor
from critic import ValueCritic, Distributional_Q_function, Distributional_V_function



def lossV1(q, v, presum_tau, expectile=0.8):
    batch_size, quantile_size = q.size()

    diff = q.unsqueeze(2) - v.unsqueeze(1)              # Shape: (batch_size, quantile_size, quantile_size)
    weight = torch.where(diff < 0, expectile, 1 - expectile)   # Shape: (batch_size, quantile_size, quantile_size)
    presum_tau_outer = presum_tau.unsqueeze(2) * presum_tau.unsqueeze(1)  # Shape: (batch_size, quantile_size, quantile_size)

    loss = weight * (diff ** 2) * presum_tau_outer  # Shape: (batch_size, quantile_size, quantile_size)
    loss = loss.sum()

    return loss



def lossQ(q, target_q, presum_tau):
    diff = target_q - q  # Shape: (N, T)

    squared_diff = diff ** 2  # Shape: (N, T)

    outer_presum_tau = presum_tau.unsqueeze(2) * presum_tau.unsqueeze(1)  # Shape: (N, T, T)

    weighted_squared_diff = squared_diff.unsqueeze(2) * outer_presum_tau  # Shape: (N, T, T)

    loss = weighted_squared_diff.sum()

    return loss



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Loss_actor(exp_a, mu, actions):
    return (exp_a.unsqueeze(-1) * ((mu - actions)**2)).mean()


def q_minus_v(q, v, presum_tau):
    presum_tau_outer = presum_tau.unsqueeze(2) * presum_tau.unsqueeze(1)  # Shape: (N, T, T)

    diff = q.unsqueeze(2) - v.unsqueeze(1)  # Shape: (N, T, T)

    weighted_diff = diff * presum_tau_outer  # Shape: (N, T, T)

    loss = weighted_diff.sum()

    return loss


class Distribution_IQL(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        n_layers,
        num_quantiles,
        expectile,
        discount,
        rate,
        temperature,
    ):
        # Actor
        self.actor = Actor(state_dim, action_dim, hidden_dim, n_layers).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=int(1e6))

        # Distributional Q function
        self.Q = Distributional_Q_function(
            state_dim, action_dim, hidden_dim, n_layers, num_quantiles
        ).to(device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=3e-4)

        # Distributional V function
        self.V = Distributional_V_function(
            state_dim, hidden_dim, n_layers, num_quantiles
        ).to(device)
        self.V_optimizer = torch.optim.Adam(self.V.parameters(), lr=3e-4)

        self.discount = discount
        self.rate = rate
        self.temperature = temperature

        self.total_it = 0
        self.expectile = expectile
        self.num_quantiles = num_quantiles


    def update_v(self, states, actions, logger=None):
        with torch.no_grad():
            tau, tau_hat, presum_tau = self.get_tau(states)
            q = self.Q_target(states, actions, tau)

        v = self.V(states, tau)
        v_loss = lossV1(q, v, presum_tau, self.expectile).mean()

        self.V_optimizer.zero_grad()
        v_loss.backward()
        self.V_optimizer.step()

        logger.log('train/value_loss', v_loss, self.total_it)
        logger.log('train/v', v.mean(), self.total_it)


    def update_q(self, states, actions, rewards, next_states, not_dones, logger=None):
        with torch.no_grad():
            next_tau, next_tau_hat, next_presum_tau = self.get_tau(next_states)
            next_v = self.V(next_states, next_tau)
            target_q = (rewards + self.discount * not_dones * next_v).detach()

        tau, _, presum_tau = self.get_tau(states)
        q1 = self.Q(states, actions, tau)
        q_loss = lossQ(q1, target_q, presum_tau).mean()

        self.Q_optimizer.zero_grad()
        q_loss.backward()
        self.Q_optimizer.step()

        logger.log('train/critic_loss', q_loss, self.total_it)
        logger.log('train/q1', q1.mean(), self.total_it)


    def update_target(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.rate * param.data + (1 - self.rate) * target_param.data)


    def update_actor(self, states, actions, logger=None):
        with torch.no_grad():
            tau, tau_hat, presum_tau = self.get_tau( states )
            v = self.V( states, tau )
            q = self.Q_target( states, actions, tau )
            exp_a = torch.exp( q_minus_v(q, v, presum_tau) * self.temperature )
            exp_a = torch.clamp( exp_a, max=100.0 ).squeeze(-1).detach()

        mu = self.actor(states)
        actor_loss = Loss_actor(exp_a, mu, actions)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_scheduler.step()

        print(f"Actor loss: {actor_loss.item()}")

        logger.log('train/actor_loss', actor_loss, self.total_it)
        logger.log('train/adv', (q - v).mean(), self.total_it)


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor.get_action(state).cpu().data.numpy().flatten()


    def get_tau(self, obs):
        presum_tau = torch.zeros(len(obs), self.num_quantiles).to(device) + 1. / self.num_quantiles
        tau = torch.cumsum(presum_tau, dim=1)

        with torch.no_grad():
            tau_hat = torch.zeros_like(tau).to(device)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau, tau_hat, presum_tau


    def train(self, replay_buffer, batch_size=256, logger=None):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Update
        self.update_v(state, action, logger)
        self.update_actor(state, action, logger)
        self.update_q(state, action, reward, next_state, not_done, logger)
        self.update_target()

    def save(self, model_dir):
        torch.save(self.critic.state_dict(), os.path.join(model_dir, f"critic_s{str(self.total_it)}.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(model_dir, f"critic_target_s{str(self.total_it)}.pth"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(
            model_dir, f"critic_optimizer_s{str(self.total_it)}.pth"))

        torch.save(self.actor.state_dict(), os.path.join(model_dir, f"actor_s{str(self.total_it)}.pth"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(
            model_dir, f"actor_optimizer_s{str(self.total_it)}.pth"))
        torch.save(self.actor_scheduler.state_dict(), os.path.join(
            model_dir, f"actor_scheduler_s{str(self.total_it)}.pth"))

        torch.save(self.value.state_dict(), os.path.join(model_dir, f"value_s{str(self.total_it)}.pth"))
        torch.save(self.value_optimizer.state_dict(), os.path.join(
            model_dir, f"value_optimizer_s{str(self.total_it)}.pth"))
