from tqdm import tqdm as _tqdm
import numpy as np
import random


def tqdm(*args, **kwargs):
    # Safety, do not overflow buffer
    return _tqdm(*args, **kwargs, mininterval=1)


def q_learning(env, policy, Q, num_episodes, discount_factor=1.0, alpha_0 = 0.5, alpha_decay=0., print_episodes=False):
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
    Q_tables = []

    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0

        if hasattr(env, 'custom_reset'):
            start_state = env.custom_reset()
        else:
            start_state = env.reset()
        done = False
        if print_episodes: print('episode', i_episode, ' - ', end='')
        while not done:
            start_action = policy.sample_action(start_state)
            if print_episodes: print(start_action, end='')
            new_state, reward, done, _ = env.step(start_action)

            Qnew = np.max(policy.Q[new_state])
            if done:
                Qnew = 0
            if alpha_decay > 0:
                alpha = alpha_0 / policy.sa_count[start_state, start_action]**(alpha_decay)
            else:
                alpha = alpha_0

            policy.Q[start_state][start_action] = \
                policy.Q[start_state][start_action] + \
                alpha * (reward + discount_factor * Qnew - policy.Q[start_state, start_action])

            start_state = new_state
            i += 1
            R += (discount_factor**i) * reward
        if print_episodes: print(' - steps:', i, 'Reward', R)
        stats.append((i, R))
        Q_tables.append(policy.Q.copy())

    episode_lengths, episode_returns = zip(*stats)
    metrics_vanilla = [episode_returns, episode_lengths]
    return policy.Q, metrics_vanilla, policy, Q_tables


def double_q_learning(env, policy, Q1, Q2, num_episodes, discount_factor=1.0, alpha_0 = 0.5, alpha_decay=0.):
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
    Q_tables = []

    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0

        if hasattr(env, 'custom_reset'):
            start_state = env.custom_reset()
        else:
            start_state = env.reset()
        done = False
        while not done:
            start_action = policy.sample_action(start_state)
            new_state, reward, done, _ = env.step(start_action)

            coinflip = random.randint(0, 1)
            if coinflip:
                policy.sa_count1[start_state,start_action]+=1
                if alpha_decay > 0:
                    alpha = alpha_0 / policy.sa_count1[start_state,start_action]**(alpha_decay)
                else:
                    alpha = alpha_0
                amax = np.argmax(policy.Q1[new_state])
                Qmax = policy.Q2[new_state, amax]
                if done:
                    Qmax = 0
                policy.Q1[start_state][start_action] = \
                    policy.Q1[start_state][start_action] \
                    + alpha*(reward + discount_factor * Qmax - policy.Q1[start_state, start_action])
            else:
                policy.sa_count2[start_state,start_action]+=1
                if alpha_decay > 0:
                    alpha = alpha_0 / policy.sa_count2[start_state,start_action]**(alpha_decay)
                else:
                    alpha = alpha_0
                amax = np.argmax(policy.Q2[new_state])
                Qmax = policy.Q1[new_state, amax]
                if done:
                    Qmax = 0
                policy.Q2[start_state][start_action] = \
                    policy.Q2[start_state][start_action] + \
                    alpha * (reward + discount_factor * Qmax - policy.Q2[start_state, start_action])
            start_state = new_state
            i += 1
            R += (discount_factor**i)*reward

        stats.append((i, R))
        Q_tables.append({"Q1": policy.Q1.copy(), "Q2": policy.Q2.copy()})

    episode_lengths, episode_returns = zip(*stats)
    metrics_double = [episode_returns,episode_lengths]
    return policy.Q1, policy.Q2, metrics_double, policy, Q_tables

