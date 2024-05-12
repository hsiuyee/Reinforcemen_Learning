# Spring 2024, 535514 Reinforcement Learning
# HW1: Policy Iteration and Value iteration for MDPs
       
import numpy as np
import gym

def get_rewards_and_transitions_from_env(env):
    # Get state and action space sizes
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Intiailize matrices
    R = np.zeros((num_states, num_actions, num_states))
    P = np.zeros((num_states, num_actions, num_states))

    # Get rewards and transition probabilitites for all transitions from an OpenAI gym environment
    for s in range(num_states):
        for a in range(num_actions):
            for transition in env.P[s][a]:
                prob, s_, r, done = transition
                R[s, a, s_] = r
                P[s, a, s_] = prob
                
    return R, P

def value_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """        
        Run value iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env: 
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for value iteration
            eps: float
                for the termination criterion of value iteration 
        ----------
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize the value function V(s)
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve V(s) using the Bellman optimality operator
            4. Derive the optimal policy using V(s)
        ----------
    """
    num_spaces = env.observation_space.n
    num_actions = env.action_space.n
    
    # Initialize with a random policy
    policy = np.array([env.action_space.sample() for _ in range(num_spaces)])
    
    ##### FINISH TODOS HERE #####
    
    # 1. Initialize the value function V(s)
    V = [0 for _ in range(num_spaces)]

    
    # 2. Get transition probabilities and reward function from the gym env
    R, P = get_rewards_and_transitions_from_env(env)

    # 3. Iterate and improve V(s) using the Bellman optimality operator
    for i in range(max_iterations):
        tempV = [0 for _ in range(num_spaces)]
        for s in range(num_spaces):
            for a in range(num_actions):
                future_reward = 0
                for s_ in range(num_spaces):
                    future_reward = future_reward + P[s][a][s_] * V[s_]
                tempV[s] = max(tempV[s],  sum(R[s][a]) + gamma * future_reward)
            
        # calculate the error
        diff = 0
        for i in range(num_spaces):
            diff = max(diff, abs(V[i] - tempV[i]))
        if (diff < eps):
            V = tempV
            break
        
        V = tempV
    
    # 4. Derive the optimal policy using V(s)
        # we find Q* first
    Q = np.zeros((num_spaces, num_actions))
    for s in range(num_spaces):
        for a in range(num_actions):
            future_reward = 0
            for s_ in range(num_spaces):
                future_reward = future_reward + P[s][a][s_] * V[s_]
            Q[s][a] = sum(R[s][a]) + gamma * future_reward
        # we find optimal policy
    for s in range(num_spaces):
        maxi = 0
        for a in range(num_actions):
            if (Q[s][maxi] < Q[s][a]):
                a = maxi
        policy[s] = maxi
        
    #############################
    
    # Return optimal policy    
    return policy

def policy_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """ 
        Run policy iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env: 
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for the policy evalaution in policy iteration
            eps: float
                for the termination criterion of policy evaluation 
        ----------  
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize with a random policy and initial value function
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve the policy
        ----------
    """
    num_spaces = env.observation_space.n
    num_actions = env.action_space.n
    
    # Initialize with a random policy
    policy = np.array([env.action_space.sample() for _ in range(num_spaces)])
    
    ##### FINISH TODOS HERE #####
    
    # 1. Initialize with a random policy and initial value function
    temp_policy = np.array([env.action_space.sample() for _ in range(num_spaces)])
    
    # 2. Get transition probabilities and reward function from the gym env
    R, P = get_rewards_and_transitions_from_env(env)
    
    # 3. Iterate and improve the policy
    # find V_pi    
    V_pi = [0 for _ in range(num_spaces)]
    for i in range(max_iterations):
        while True:
            tempV_pi = [0 for _ in range(num_spaces)]
            for s in range(num_spaces):
                future_reward = 0
                for s_ in range(num_spaces):
                    future_reward = future_reward + P[s][policy[s]][s_] * V_pi[s_]
                
                tempV_pi[s] = (sum(R[s][policy[s]]) + gamma * future_reward)            # calculate the error
            diff = 0
            for i in range(num_spaces):
                diff = max(diff, abs(V_pi[i] - tempV_pi[i]))
            if (diff < eps):
                V_pi = tempV_pi
                break
            
            V_pi = tempV_pi
            
        # find_Q_pi
        Q_pi = np.zeros((num_spaces, num_actions))
        for s in range(num_spaces):
            for a in range(num_actions):
                future_reward = 0
                for s_ in range(num_spaces):
                    future_reward = future_reward + P[s][a][s_] * V_pi[s_]
                Q_pi[s][a] = sum(R[s][a]) + gamma * future_reward
                
        # we find optimal policy
        for s in range(num_spaces):
            maxi = 0
            for a in range(num_actions):
                if (Q_pi[s][maxi] < Q_pi[s][a]):
                    a = maxi
            policy[s] = maxi

        flag = 0
        for s in range(num_spaces):
            if temp_policy[s] != policy[s]:
                flag = True
        if flag:
            policy = temp_policy
        else:
            break
    
    #############################

    # Return optimal policy
    return policy

def print_policy(policy, mapping=None, shape=(0,)):
    print(np.array([mapping[action] for action in policy]).reshape(shape))


def run_pi_and_vi(env_name):
    """ 
        Enforce policy iteration and value iteration
    """    
    env = gym.make(env_name)
    print('== {} =='.format(env_name))
    print('# of actions:', env.action_space.n)
    print('# of states:', env.observation_space.n)
    print(env.desc)

    vi_policy = value_iteration(env)
    pi_policy = policy_iteration(env)

    return pi_policy, vi_policy


if __name__ == '__main__':
    # OpenAI gym environment: Taxi-v2 or Taxi-v3
    pi_policy, vi_policy = run_pi_and_vi('Taxi-v3')

    # For debugging
    action_map = {0: "S", 1: "N", 2: "E", 3: "W", 4: "P", 5: "D"}
    print_policy(pi_policy, action_map, shape=None)
    print_policy(vi_policy, action_map, shape=None)
    
    # Compare the policies obatined via policy iteration and value iteration
    diff = sum([abs(x-y) for x, y in zip(pi_policy.flatten(), vi_policy.flatten())])        
    print('Discrepancy:', diff)

