from tqdm import tqdm as _tqdm
import numpy as np
import random

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer


def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> state-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        start_state = env.reset()
        start_action = policy.sample_action(start_state)
        done = False
        while not done:
            new_step = env.step(start_action)
            new_state = new_step[0]
            reward = new_step[1]
            done = new_step[2]
            new_action = policy.sample_action(new_state)
            policy.Q[start_state][start_action] = policy.Q[start_state][start_action] + alpha*(reward + \
                                            discount_factor*policy.Q[new_state,new_action] - policy.Q[start_state,start_action])
            #print(Q)
            start_state = new_state
            start_action = new_action
            i+=1
            R+=(discount_factor**i)*reward
            
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns), policy


def q_learning(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        policy: A behavior policy which allows us to sample actions with its sample_action method.
        Q: Q value function
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        start_state = env.reset()
        done = False
        while not done:
            start_action = policy.sample_action(start_state)
            new_step = env.step(start_action)
            new_state = new_step[0]
            reward = new_step[1]
            done = new_step[2]
            policy.Q[start_state][start_action] = policy.Q[start_state][start_action] + alpha*(reward + \
                                            discount_factor*np.max(policy.Q[new_state]) - policy.Q[start_state,start_action])
            #print(Q)
            start_state = new_state
            i+=1
            R+=(discount_factor**i)*reward
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns), policy

def double_q_learning(env, policy, Q1, Q2, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        policy: A behavior policy which allows us to sample actions with its sample_action method.
        Q: Q value function
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        start_state = env.reset()
        done = False
        while not done:
            start_action = policy.sample_action(start_state)
            new_step = env.step(start_action)
            new_state = new_step[0]
            reward = new_step[1]
            done = new_step[2]
            coinflip = random.randint(0,1)
            if coinflip:
                policy.Q1[start_state][start_action] = policy.Q1[start_state][start_action] + alpha*(reward + \
                                            discount_factor*np.max(policy.Q2[new_state]) - policy.Q1[start_state,start_action])
            else:
                policy.Q2[start_state][start_action] = policy.Q2[start_state][start_action] + alpha*(reward + \
                                            discount_factor*np.max(policy.Q1[new_state]) - policy.Q2[start_state,start_action])
            start_state = new_state
            i+=1
            R+=(discount_factor**i)*reward
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q1, Q2, (episode_lengths, episode_returns), policy