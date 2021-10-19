from itertools import product
import numpy as np


def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    def _coord_to_state(r, c):
        return env.observation([r, c])

    def _state_to_coord(state):
        c = state % env.cols
        r = (state - c) // env.cols
        return r, c

    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))

    err = np.inf
    while err > theta:
        err = 0
        for s, a in product(range(env.nS), range(env.nA)):
            coord = tuple(_state_to_coord(s))

            if coord == env.goal:
                new_q = env.final_reward
            else:

                new_q = sum(
                    [
                        prob
                        * (
                            reward
                            + discount_factor
                            * np.max(Q[_coord_to_state(*next_coord), :])
                        )
                        for prob, next_coord, reward, done in env.P[coord][a]
                    ]
                )

            err = max(err, abs(new_q - Q[s, a]))
            Q[s, a] = new_q

    Q_coord = {tuple(_state_to_coord(s)): q_vals for s, q_vals in enumerate(Q)}
    return Q, Q_coord
