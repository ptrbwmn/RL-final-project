import numpy as np
from collections import defaultdict

import sys
import random
import time
from policy import EpsilonGreedyPolicy, EpsilonGreedyPolicy_Double_Q
from q_learning import sarsa, q_learning, double_q_learning
from plotting import plot

from env_hasselt4_1 import BasicEnv2
env = BasicEnv2()

class egPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        epsilon = self.epsilon
        greedy = np.random.choice(2,1,p=[epsilon, 1-epsilon])
        
        if obs == 0:
            num_actions = 2
            if greedy:
                action = np.argmax(self.Q[obs,:2])
            else:
                action = np.random.choice(num_actions,1,p=np.ones(num_actions)/num_actions)[0]
        else:
            num_actions = 10
            if greedy:
                action = 2+np.argmax(self.Q[obs,2:])
            else:
                action = 2+np.random.choice(num_actions,1,p=np.ones(num_actions)/num_actions)[0]
        return action

class EpsilonGreedyPolicy_Double_Q(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q1, Q2, epsilon):
        self.Q1 = Q1
        self.Q2 = Q2
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        epsilon = self.epsilon
        num_actions = self.Q1.shape[1]
        greedy = np.random.choice(2,1,p=[epsilon, 1-epsilon])
        if greedy:
            action = np.argmax(self.Q1[obs]+self.Q2[obs])
        else:
            action = np.random.choice(num_actions,1,p=np.ones(num_actions)/num_actions)[0]
        return action

Q=np.ones((2,12))
policy=egPolicy(Q,0.5)
s=env.reset()

Q_q_learning, (episode_lengths_q_learning, episode_returns_q_learning), policy_q_learning = q_learning(env, policy, Q, 1000)

plot(episode_lengths_q_learning, episode_returns_q_learning, "q_learning")

k=0