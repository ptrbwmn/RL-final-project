from gym_minigrid.register import register
import matplotlib.pyplot as plt
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.envs.empty import EmptyEnv
import numpy as np
from collections import defaultdict
import sys
import random
import time
from policy import EpsilonGreedyPolicy, EpsilonGreedyPolicy_Double_Q
from q_learning import sarsa, q_learning, double_q_learning
#from plotting import plot
from tqdm import tqdm as _tqdm
import numpy as np
import random

from env_dense import EmptyEnvDense5x5


def tqdm(*args, **kwargs):
    # Safety, do not overflow buffer
    return _tqdm(*args, **kwargs, mininterval=1)


env = gym.make('MiniGrid-EmptyDense-5x5-v0')

obs = env.reset()  # This now produces an RGB tensor only
env.render()
input('press any key to start')
nS = env.observation_space.spaces.__sizeof__()
# Q={}
# if env.hash() not in Q:
#     Q[env.hash()] = (0,0,0)
Q = np.zeros((nS, 3))
Q = np.ones((nS, 3))


def q_learning_(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
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
    HT = {}
    for i_episode in range(num_episodes):  # tqdm(range(num_episodes)):
        i = 0
        R = 0
        print('episode', i_episode, '-', end='')
        start_state = env.reset()
        if env.hash() not in HT:
            HT[env.hash()] = len(HT.keys())

        done = False
        while not done:
            start_state = HT[env.hash()]
            start_action = policy.sample_action(start_state)
            print(start_action, end='')
            new_step = env.step(start_action)
            new_state = new_step[0]
            if env.hash() not in HT:
                HT[env.hash()] = len(HT.keys())
            new_state = HT[env.hash()]
            reward = new_step[1]
            done = new_step[2]
            Qnew = np.max(policy.Q[new_state])
            if done:
                Qnew = 0
            policy.Q[start_state][start_action] = policy.Q[start_state][start_action] + alpha*(reward +
                                                                                               discount_factor*Qnew - policy.Q[start_state, start_action])
            # print(Q)
            start_state = new_state
            i += 1
            R += (discount_factor**i)*reward

        stats.append((i, R))
        print(', steps:', i, ', reward:', R)
    episode_lengths, episode_returns = zip(*stats)
    return [Q], episode_returns, policy, episode_lengths


policy = EpsilonGreedyPolicy(Q, 0.1)

Q_q_learning1, episode_returns1, policy_q_learning1, episode_lengths1 = q_learning_(
    env, policy, Q, 60)
plt.plot(episode_lengths1)
plt.show()
input('press key to end')
