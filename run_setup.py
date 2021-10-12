from environments import WindyGridworldEnv, BasicEnv2
from policy import EpsilonGreedyPolicy, EpsilonGreedyPolicy_Double_Q
from q_learning import sarsa, q_learning, double_q_learning

import numpy as np


def run_setup(config, q_learning_variant):  
    policy = config['policy']
    epsilon = config['epsilon']
    gamma = config['gamma']
    alpha = config['alpha']
    num_iter = config['num_iter']

    env = config['env']
    if env == "WindyGridworldEnv":
        env = WindyGridworldEnv()
    elif env == "BasicEnv2":
        env = BasicEnv2()
    else:
        raise NotImplementedError
    
    if policy == "EpsilonGreedy":
        if q_learning_variant == "vanilla":
            Q = np.zeros((env.nS, env.nA))
            policy = EpsilonGreedyPolicy(Q, epsilon=epsilon)
            Q_table, episode_returns, policy = q_learning(env, policy, Q, num_iter, discount_factor=gamma, alpha=alpha)
            return Q_table, episode_returns, policy
        elif q_learning_variant == "double":
            Q1 = np.zeros((env.nS, env.nA))
            Q2 = np.zeros((env.nS, env.nA))
            policy = EpsilonGreedyPolicy_Double_Q(Q1, Q2, epsilon=epsilon)
            Q_table1, Q_table2, episode_returns, policy = double_q_learning(env, policy, Q1, Q2, num_iter,  discount_factor=gamma, alpha=alpha)
            return Q_table1, Q_table2, episode_returns, policy
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError