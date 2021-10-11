import numpy as np
from collections import defaultdict

import sys
import random
import time
from policy import EpsilonGreedyPolicy, EpsilonGreedyPolicy_Double_Q
from q_learning import sarsa, q_learning, double_q_learning
from plotting import plot

from windy_gridworld import WindyGridworldEnv
env = WindyGridworldEnv()


# Q = np.zeros((env.nS, env.nA))
# policy = EpsilonGreedyPolicy(Q, epsilon=0.1)
# Q_sarsa, (episode_lengths_sarsa, episode_returns_sarsa), policy_sarsa = sarsa(env, policy, Q, 10000)

# plot(episode_lengths_sarsa, episode_returns_sarsa, "sarsa")

Q = np.zeros((env.nS, env.nA))
policy = EpsilonGreedyPolicy(Q, epsilon=0.1)
Q_q_learning, (episode_lengths_q_learning, episode_returns_q_learning), policy_q_learning = q_learning(env, policy, Q, 10000)

plot(episode_lengths_q_learning, episode_returns_q_learning, "q_learning")

Q1 = np.zeros((env.nS, env.nA))
Q2 = np.zeros((env.nS, env.nA))
policy = EpsilonGreedyPolicy_Double_Q(Q1, Q2, epsilon=0.1)
Q1_double_q_learning, Q2_double_q_learning, (episode_lengths_double_q_learning, episode_returns_double_q_learning), policy_double_q_learning = double_q_learning(env, policy, Q1, Q2, 10000)

plot(episode_lengths_double_q_learning, episode_returns_double_q_learning, "double_q_learning")