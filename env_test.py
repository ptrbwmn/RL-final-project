import numpy as np
from collections import defaultdict

import sys
import random
import time
from policy import EpsilonGreedyPolicy, EpsilonGreedyPolicy_Double_Q
from q_learning import sarsa, q_learning, double_q_learning
#from plotting import plot

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
                maxx=np.max(self.Q[obs,:2])
                test=np.where(self.Q[obs,:2]==maxx)
                action = np.random.choice(test[0])
                #action = np.argmax(self.Q[obs,:2])
            else:
                action = np.random.choice(num_actions,1,p=np.ones(num_actions)/num_actions)[0]
        else:
            num_actions = 10
            if greedy:
                maxx=np.max(self.Q[obs,2:])
                test=np.where(self.Q[obs,2:]==maxx)
                action = 2+np.random.choice(test[0])
                #action = 2+np.argmax(self.Q[obs,2:])
            else:
                action = 2+np.random.choice(num_actions,1,p=np.ones(num_actions)/num_actions)[0]
        return action

class egPolicy_Double_Q(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q1, Q2, epsilon):
        self.Q1 = Q1.copy()
        self.Q2 = Q2.copy()
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
        #num_actions = self.Q1.shape[1]
        greedy = np.random.choice(2,1,p=[epsilon, 1-epsilon])
        
        if obs == 0:
            num_actions = 2
            if greedy:
                comb = self.Q1+self.Q2
                maxx=np.max(comb[obs,:2])
                test=np.where(comb[obs,:2]==maxx)
                action = np.random.choice(test[0])
                #action = np.argmax(self.Q1[obs,:2]+self.Q2[obs,:2])
            else:
                action = np.random.choice(num_actions,1,p=np.ones(num_actions)/num_actions)[0]
        else:
            num_actions = 10
            if greedy:
                comb = self.Q1+self.Q2
                maxx=np.max(comb[obs,2:])
                test=np.where(comb[obs,2:]==maxx)
                action = 2+np.random.choice(test[0])
                #action = 2+np.argmax(self.Q1[obs,2:]+self.Q2[obs,2:])
            else:
                action = 2+np.random.choice(num_actions,1,p=np.ones(num_actions)/num_actions)[0]
        return action

K1=np.zeros(300).astype(np.float)
K2=np.zeros(300).astype(np.float)
runs=100
for turns in range(runs):
    Q=np.ones((2,12))
    #Q=np.zeros((2,12))
    Q[1,0]=-1e6
    Q[1,1]=-1e6
    s=env.reset()

    policy1=egPolicy(Q,0.1)
    Q_q_learning1, episode_returns1, policy_q_learning1, episode_lengths1 = q_learning(env, policy1, Q, 300)

    #Q=np.ones((2,12))
    Q=np.zeros((2,12))
    Q[1,0]=-1e6
    Q[1,1]=-1e6
    s=env.reset()
    policy2=egPolicy_Double_Q(Q,Q,0.1)
    Q_q_learning2, episode_returns2, policy_q_learning2, episode_lengths2 = double_q_learning(env, policy2, Q, Q, 300)

    e1=np.array(episode_lengths1)-1
    k1=e1.astype(np.float)
    cum=0
    for i in range(e1.shape[0]):
        cum+=e1[i]
        k1[i]=cum/(i+1)

    e2=np.array(episode_lengths2)-1
    k2=e2.astype(np.float)
    cum=0
    for i in range(e2.shape[0]):
        cum+=e2[i]
        k2[i]=cum/(i+1)
    K1+=k1
    K2+=k2
K1/=runs
K2/=runs
    
import matplotlib.pyplot as plt
#plt.plot(episode_returns)
#plt.show()

plt.plot(K1,label='q')

plt.plot(K2,label='doubleq')
plt.legend()

plt.show()