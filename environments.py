import sys
from copy import deepcopy
from itertools import product

import gym
import numpy as np
from gym import error, spaces, utils
from gym.envs.toy_text import discrete
from gym.utils import seeding

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class WindyGridworldEnv(discrete.DiscreteEnv):

    metadata = {"render.modes": ["human", "ansi"]}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        new_position = (
            np.array(current)
            + np.array(delta)
            + np.array([-1, 0]) * winds[tuple(current)]
        )
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == (3, 7)
        return [(1.0, new_state, -1.0, is_done)]

    def __init__(self):
        self.shape = (7, 10)

        nS = np.prod(self.shape)
        nA = 4

        # Wind strength
        winds = np.zeros(self.shape)
        winds[:, [3, 4, 5, 8]] = 1
        winds[:, [6, 7]] = 2

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)

        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3, 0), self.shape)] = 1.0

        super(WindyGridworldEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode="human", close=False):
        self._render(mode, close)

    def _render(self, mode="human", close=False):
        if close:
            return

        outfile = StringIO() if mode == "ansi" else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " x "
            elif position == (3, 7):
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")


class BasicEnv2(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        #
        self.nS = 2
        self.nA = 12
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        self.state = 0

    def step(self, action):

        #
        if self.state == 1:
            reward = np.random.normal(-0.1, 1)
            done = True
        elif self.state == 0:
            if action == 0:
                reward = 0
                done = True
            elif action == 1:
                reward = 0
                self.state = 1
                done = False
            else:
                print("ERROR")
        info = {}
        state = self.state
        return state, reward, done, info

    def reset(self):
        self.state = 0
        return self.state

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class CliffWalkingEnv(gym.Env):
    """Cliff Walking Environment
    See the README.md file from https://github.com/caburu/gym-cliffwalking
    """

    # There is no renderization yet
    # metadata = {'render.modes': ['human']}

    def observation(self, state):
        return state[0] * self.cols + state[1]

    def __init__(self):
        self.nS = 48
        self.nA = 4
        self.rows = 4
        self.cols = 12
        self.start = [0, 0]
        self.goal = [0, 11]
        self.current_state = None

        # There are four actions: up, down, left and right
        self.action_space = spaces.Discrete(4)

        # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Discrete(self.rows * self.cols)

    def step(self, action):
        new_state = deepcopy(self.current_state)

        if action == 1:  # right
            new_state[1] = min(new_state[1] + 1, self.cols - 1)
        elif action == 2:  # down
            new_state[0] = max(new_state[0] - 1, 0)
        elif action == 3:  # left
            new_state[1] = max(new_state[1] - 1, 0)
        elif action == 0:  # up
            new_state[0] = min(new_state[0] + 1, self.rows - 1)
        else:
            raise Exception("Invalid action.")
        self.current_state = new_state

        reward = -1.0
        is_terminal = False
        if self.current_state[0] == 0 and self.current_state[1] > 0:
            if self.current_state[1] < self.cols - 1:
                reward = -100.0
                self.current_state = deepcopy(self.start)
            else:
                is_terminal = True

        return self.observation(self.current_state), reward, is_terminal, {}

    def reset(self):
        self.current_state = self.start
        return self.observation(self.current_state)

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def get_state_color(self, state_number):
        if state_number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            return "grey"
        elif state_number == 11:
            return "green"
        elif state_number == 0:
            return "blue"
        else:
            return "white"


class BaseGrid(gym.Env):
    """"""

    # There is no renderization yet
    # metadata = {'render.modes': ['human']}

    def __init__(
        self,
        nS,
        nA,
        rows,
        cols,
        start,
        goal,
        final_reward,
        lava_cells,
        p_forward,
        stoch_reward=False,
    ):
        self.nS = nS
        self.nA = nA
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.final_reward = final_reward
        self.lava_cells = lava_cells
        self.p_forward = p_forward
        self.stoch_reward = stoch_reward

        self.current_state = None

        # There are four actions: up, down, left and right
        self.action_space = spaces.Discrete(4)

        # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Discrete(self.rows * self.cols)

        # transition_probabilities
        self.P = self._get_transitions()

    def observation(self, state):
        return state[0] * self.cols + state[1]

    def get_state_color(self, state_number):
        if state_number in []:
            return "grey"  # cliff
        elif state_number in [self.observation(cell) for cell in self.lava_cells]:
            return "orange"  # lava
        elif state_number == self.observation(self.start):
            return "blue"  # start position
        elif state_number == self.observation(self.goal):
            return "green"  # goal position
        else:
            return "white"

    def _new_loc(self, r, c, action):
        if action == UP:  # 0
            r2 = min(r + 1, self.rows - 1)
            return [r2, c]
        elif action == RIGHT:  # 1
            c2 = min(c + 1, self.cols - 1)
            return [r, c2]
        elif action == DOWN:  # 2
            r2 = max(r - 1, 0)
            return [r2, c]
        elif action == LEFT:  # 3
            c2 = max(c - 1, 0)
            return [r, c2]
        else:
            raise Exception("Invalid action.")

    def _get_transitions(self):
        """transition matrix"""
        # initialise
        transitions = {
            (r, c): {a: [] for a in range(self.nA)}
            for r, c in product(range(self.rows), range(self.cols))
        }

        # loop over state/action pairs
        for r, c, a in product(range(self.rows), range(self.cols), range(self.nA)):

            possible_a = [(a + d) % 4 for d in [-1, 0, 1]]
            prob_side = (1 - self.p_forward) / 2
            probs = [prob_side, self.p_forward, prob_side]
            new_locs = [self._new_loc(r, c, pa) for pa in possible_a]

            if self.stoch_reward:
                rewards = [-12, 10]
            else:
                rewards = [-1]

            for new_loc, loc_prob in zip(new_locs, probs):
                for reward in rewards:
                    done = False
                    if new_loc in self.lava_cells:
                        reward = -1000
                        done = True
                    if new_loc == self.goal:
                        reward = self.final_reward
                        done = True
                    transitions[(r, c)][a].append(
                        (loc_prob / len(rewards), new_loc, reward, done)
                    )
        return transitions

    def step(self, action):
        next_steps = self.P[tuple(self.current_state)][action]
        probs = [p for p, _, _, _ in next_steps]
        chosen = np.random.choice(range(len(next_steps)), p=probs)
        _, new_state, reward, is_terminal = next_steps[chosen]

        self.current_state = new_state
        return self.observation(new_state), reward, is_terminal, {}

    def reset(self):
        self.current_state = self.start
        return self.observation(self.current_state)

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class LavaWorld5x7Determ(BaseGrid):
    """"""

    def __init__(self):
        super().__init__(
            nS=35,
            nA=4,
            rows=5,
            cols=7,
            start=[4, 0],
            goal=[4, 6],
            final_reward=8,
            lava_cells=[[4, 2], [4, 3], [4, 4], [0, 2], [0, 3], [0, 4]],
            p_forward=1.0,
            stoch_reward=False,
        )


class LavaWorld5x7StochMovement(BaseGrid):
    """"""

    def __init__(self):
        super().__init__(
            nS=35,
            nA=4,
            rows=5,
            cols=7,
            start=[4, 0],
            goal=[4, 6],
            final_reward=10,
            lava_cells=[[4, 2], [4, 3], [4, 4], [0, 2], [0, 3], [0, 4]],
            p_forward=0.8,
            stoch_reward=False,
        )


class LavaWorld5x7StochRewards(BaseGrid):
    """"""

    def __init__(self):
        super().__init__(
            nS=35,
            nA=4,
            rows=5,
            cols=7,
            start=[4, 0],
            goal=[4, 6],
            final_reward=10,
            lava_cells=[[4, 2], [4, 3], [4, 4], [0, 2], [0, 3], [0, 4]],
            p_forward=1,
            stoch_reward=True,
        )


class LavaWorld13x15Determ(BaseGrid):
    """"""

    def __init__(self):
        super().__init__(
            nS=195,
            nA=4,
            rows=13,
            cols=15,
            start=[12, 0],
            goal=[12, 14],
            final_reward=26,
            lava_cells=[
                [12, 2],
                [12, 3],
                [12, 4],
                [12, 5],
                [12, 6],
                [12, 7],
                [12, 8],
                [12, 9],
                [12, 10],
                [12, 11],
                [12, 12],
                [11, 7],
                [10, 7],
                [9, 7],
                [8, 2],
                [8, 3],
                [8, 4],
                [8, 5],
                [8, 6],
                [8, 7],
                [8, 8],
                [8, 9],
                [8, 10],
                [8, 11],
                [8, 12],
                [7, 7],
                [5, 7],
                [4, 2],
                [4, 3],
                [4, 4],
                [4, 5],
                [4, 6],
                [4, 7],
                [4, 8],
                [4, 9],
                [4, 10],
                [4, 11],
                [4, 12],
                [0, 2],
                [0, 3],
                [0, 4],
                [0, 5],
                [0, 6],
                [0, 7],
                [0, 8],
                [0, 9],
                [0, 10],
                [0, 11],
                [0, 12],
            ],
            p_forward=1.0,
            stoch_reward=False,
        )


class LavaWorld13x15StochMovement(BaseGrid):
    """"""

    def __init__(self):
        super().__init__(
            nS=195,
            nA=4,
            rows=13,
            cols=15,
            start=[12, 0],
            goal=[12, 14],
            final_reward=34,
            lava_cells=[
                [12, 2],
                [12, 3],
                [12, 4],
                [12, 5],
                [12, 6],
                [12, 7],
                [12, 8],
                [12, 9],
                [12, 10],
                [12, 11],
                [12, 12],
                [11, 7],
                [10, 7],
                [9, 7],
                [8, 2],
                [8, 3],
                [8, 4],
                [8, 5],
                [8, 6],
                [8, 7],
                [8, 8],
                [8, 9],
                [8, 10],
                [8, 11],
                [8, 12],
                [7, 7],
                [5, 7],
                [4, 2],
                [4, 3],
                [4, 4],
                [4, 5],
                [4, 6],
                [4, 7],
                [4, 8],
                [4, 9],
                [4, 10],
                [4, 11],
                [4, 12],
                [0, 2],
                [0, 3],
                [0, 4],
                [0, 5],
                [0, 6],
                [0, 7],
                [0, 8],
                [0, 9],
                [0, 10],
                [0, 11],
                [0, 12],
            ],
            p_forward=0.8,
            stoch_reward=False,
        )


class LavaWorld13x15StochRewards(BaseGrid):
    """"""

    def __init__(self):
        super().__init__(
            nS=195,
            nA=4,
            rows=13,
            cols=15,
            start=[12, 0],
            goal=[12, 14],
            final_reward=34,
            lava_cells=[
                [12, 2],
                [12, 3],
                [12, 4],
                [12, 5],
                [12, 6],
                [12, 7],
                [12, 8],
                [12, 9],
                [12, 10],
                [12, 11],
                [12, 12],
                [11, 7],
                [10, 7],
                [9, 7],
                [8, 2],
                [8, 3],
                [8, 4],
                [8, 5],
                [8, 6],
                [8, 7],
                [8, 8],
                [8, 9],
                [8, 10],
                [8, 11],
                [8, 12],
                [7, 7],
                [5, 7],
                [4, 2],
                [4, 3],
                [4, 4],
                [4, 5],
                [4, 6],
                [4, 7],
                [4, 8],
                [4, 9],
                [4, 10],
                [4, 11],
                [4, 12],
                [0, 2],
                [0, 3],
                [0, 4],
                [0, 5],
                [0, 6],
                [0, 7],
                [0, 8],
                [0, 9],
                [0, 10],
                [0, 11],
                [0, 12],
            ],
            p_forward=1,
            stoch_reward=True,
        )


class SimpleWorld3x3Determ(gym.Env):
    """"""

    def __init__(self):
        super().__init__(
            nS=9,
            nA=4,
            rows=3,
            cols=3,
            start=[2, 0],
            goal=[0, 2],
            final_reward=10,
            lava_cells=[],
            p_forward=1,
            stoch_reward=False,
        )


class SimpleWorld3x3StochMovement(BaseGrid):
    """"""

    def __init__(self):
        super().__init__(
            nS=9,
            nA=4,
            rows=3,
            cols=3,
            start=[2, 0],
            goal=[0, 2],
            final_reward=10,
            lava_cells=[],
            p_forward=0.8,
            stoch_reward=False,
        )


class SimpleWorld3x3StochRewards(BaseGrid):
    """"""

    def __init__(self):
        super().__init__(
            nS=9,
            nA=4,
            rows=3,
            cols=3,
            start=[2, 0],
            goal=[0, 2],
            final_reward=5,
            lava_cells=[],
            p_forward=1,
            stoch_reward=True,
        )

        # TODO: why is final_reward 5 here?
