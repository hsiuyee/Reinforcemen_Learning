# Spring 2024, 535514 Reinforcement Learning
# HW2: REINFORCE and baseline

import os
import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define a tensorboard writer
writer = SummaryWriter(save_path)

class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__()

        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        self.double()

        ########## YOUR CODE HERE (5~10 lines) ##########

        # fully connected layer
        self.shared_layer = nn.Linear(self.observation_dim, self.hidden_size)

        # pi_theta(s, a)
        self.action_layer = nn.Linear(self.hidden_size, self.action_dim)

        # V^pi_theta(s)
        self.value_layer = nn.Linear(self.hidden_size, 1)

        # Random weight initialization
        init.xavier_uniform_(self.shared_layer.weight)
        init.xavier_uniform_(self.action_layer.weight)
        init.xavier_uniform_(self.value_layer.weight)

        ########## END OF YOUR CODE ##########

        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        ########## YOUR CODE HERE (3~5 lines) ##########
        state = self.shared_layer(state)
        state = F.relu(state)

        # action_prob = pi_theta(a|s)
        action_prob = self.action_layer(state)

        # V^pi_theta(s)
        state_value = self.value_layer(state)

        ########## END OF YOUR CODE ##########

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        ########## YOUR CODE HERE (3~5 lines) ##########
        # linear transformation
        state = torch.Tensor(state)

        # use forward function to get
        # pi_theta(s, a | s), V^pi_theta(s), given state s
        action, state_value= self.forward(state)

        # get the distribution
        m = Categorical(logits=action)

        # get sample probility
        action = m.sample()

        ########## END OF YOUR CODE ##########

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.999):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """

        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        loss = []
        value_losses = []
        returns = []

        ########## YOUR CODE HERE (8-15 lines) ##########
        # Calculate rewards-to-go
        discounted_sum = 0
        for reward in reversed(self.rewards):
            discounted_sum = reward + gamma * discounted_sum
            returns.append(discounted_sum)

        returns.reverse()
        returns = torch.Tensor(returns)

        # Get probability and value
        log_probs = [action.log_prob for action in saved_actions]
        action_log_probs = torch.stack(log_probs, dim=0)

        # Calculate policy loss using policy gradient
        policy_loss = -(returns * action_log_probs).sum()

        # Calculate value loss using MSE loss
        values = [value.value for value in saved_actions]
        value_predictions = torch.stack(values, dim=0).squeeze(1)
        value_targets = returns.detach()
        value_loss = F.mse_loss(value_predictions, value_targets)

        # Total loss is the sum of policy loss and value loss
        loss = policy_loss + value_loss

        ########## END OF YOUR CODE ##########

        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

'''
class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps          # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done):
    """
        Implement Generalized Advantage Estimation (GAE) for your value prediction
        TODO (1): Pass correct corresponding inputs (rewards, values, and done) into the function arguments
        TODO (2): Calculate the Generalized Advantage Estimation and return the obtained value
    """

        ########## YOUR CODE HERE (8-15 lines) ##########




        ########## END OF YOUR CODE ##########
'''

def train(lr=0.01):
    """
        Train the model using SGD (via backpropagation)
        TODO (1): In each episode,
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode

        TODO (2): In each episode,
        1. record all the value you aim to visualize on tensorboard (lr, reward, length, ...)
    """

    # Instantiate the policy model and the optimizer
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler (optional)
    # scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # EWMA reward for tracking the learning progress
    ewma_reward = 0

    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0

        # Uncomment the following line to use learning rate scheduler
        # scheduler.step()

        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process

        ########## YOUR CODE HERE (10-15 lines) ##########

        for t in range(9999):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        optimizer.zero_grad()
        loss = model.calculate_loss()
        loss.backward()
        optimizer.step()
        model.clear_memory()

        ########## END OF YOUR CODE ##########

        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        if i_episode % 20 == 0:
            print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        #Try to use Tensorboard to record the behavior of your implementation

        ########## YOUR CODE HERE (4-5 lines) ##########

        writer.add_scalar('reward', ep_reward, i_episode)
        writer.add_scalar('ep_reward', ewma_reward, i_episode)
        writer.add_scalar('policy_loss', loss, i_episode)
        writer.add_scalar('episode_length', t, i_episode)
        # writer.add_scalar('learning_rate', lr, i_episode)

        ########## END OF YOUR CODE ##########

        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            torch.save(model.state_dict(), save_path + '/CartPole_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, n_episodes=10):
    """
        Test the learned model (no change needed)
    """
    model = Policy()

    model.load_state_dict(torch.load('/content/drive/My Drive/資訊工程學習資料/強化學習原理/課程作業 (謝秉均)/HW2/reinforce/{}'.format(name)))

    render = True
    max_episode_len = 10000

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()


if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10
    lr = 0.01
    env = gym.make('CartPole-v0')
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    train(lr)
    test(f'CartPole_{lr}.pth')