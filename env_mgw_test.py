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

# obs = env.reset()  # This now produces an RGB tensor only
obs = env.custom_reset()
env.render()
input('press any key to start')
nS = env.observation_space.spaces.__sizeof__()
# Q={}
# if env.hash() not in Q:
#     Q[env.hash()] = (0,0,0)
Q = np.zeros((nS, 3))
Q = np.ones((nS, 3))


policy = EpsilonGreedyPolicy(Q, 0.1)

Q_q_learning1, episode_returns1, policy_q_learning1, episode_lengths1 = q_learning(
    env, policy, Q, 60)
plt.plot(episode_lengths1)
plt.show()
input('press key to end')
