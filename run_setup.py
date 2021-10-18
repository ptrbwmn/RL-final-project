from environments import WindyGridworldEnv, BasicEnv2, CliffWalkingEnv, LavaWorld5x7Determ, LavaWorld5x7StochMovement, LavaWorld13x15Determ, LavaWorld13x15StochMovement, SimpleWorld3x3Determ, SimpleWorld3x3StochMovement, SimpleWorld3x3StochRewards
from env_dense import EmptyEnvDense5x5
from env_lava_det import LavaDetEnv9x7
from env_lava_stoch import LavaStoch80Env9x7
from policy import EpsilonGreedyPolicy, EpsilonGreedyPolicy_Double_Q
from q_learning import q_learning, double_q_learning
import gym
import numpy as np


def run_setup(config, q_learning_variant):  
    policy = config['policy']
    epsilon = config['epsilon']
    eps_decay = config['eps_decay']
    gamma = config['gamma']
    alpha = config['alpha']
    num_iter = config['num_iter']

    env = config['env']
    if env == "WindyGridworldEnv":
        env = WindyGridworldEnv()
    elif env == "EmptyDenseEnv5x5":
        env = gym.make('MiniGrid-EmptyDense-5x5-v0')
    elif env == "CliffWalkingEnv":
        env = CliffWalkingEnv()
    elif env == "LavaDetEnv9x7":
        env = gym.make('MiniGrid-LavaDet-9x7-v0')
    elif env == "LavaStoch80Env9x7":
        env = gym.make('MiniGrid-LavaStoch80-9x7-v0')
    elif env == "LavaWorld5x7Determ":
        env = LavaWorld5x7Determ()
    elif env == "LavaWorld5x7StochMovement":
        env = LavaWorld5x7StochMovement()
    elif env == "LavaWorld13x15Determ":
        env = LavaWorld13x15Determ()
    elif env == "LavaWorld13x15StochMovement":
        env = LavaWorld13x15StochMovement()
    elif env == "SimpleWorld3x3Determ":
        env = SimpleWorld3x3Determ()
    elif env == "SimpleWorld3x3StochMovement":
        env = SimpleWorld3x3StochMovement()
    elif env == "SimpleWorld3x3StochRewards":
        env = SimpleWorld3x3StochRewards()
    else:
        raise NotImplementedError
    
    if policy == "EpsilonGreedy":
        if q_learning_variant == "vanilla":
            Q = np.zeros((env.nS, env.nA))
            policy = EpsilonGreedyPolicy(Q, epsilon, eps_decay)
            Q_table, metrics, policy, Q_tables = q_learning(env, policy, Q, num_iter, discount_factor=gamma, alpha=alpha)
            return Q_table, np.array(metrics), policy, Q_tables, env
        elif q_learning_variant == "double":
            Q1 = np.zeros((env.nS, env.nA))
            Q2 = np.zeros((env.nS, env.nA))
            policy = EpsilonGreedyPolicy_Double_Q(Q1, Q2, epsilon, eps_decay)
            Q_table1, Q_table2, metrics, policy, Q_tables= double_q_learning(env, policy, Q1, Q2, num_iter,  discount_factor=gamma, alpha=alpha)
            return Q_table1, Q_table2, np.array(metrics), policy, Q_tables, env
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError