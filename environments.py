import gym
import numpy as np
import sys
from gym.envs.toy_text import discrete

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from copy import deepcopy 

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class WindyGridworldEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]
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
        winds[:,[3,4,5,8]] = 1
        winds[:,[6,7]] = 2

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)

        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3,0), self.shape)] = 1.0

        super(WindyGridworldEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human', close=False):
        self._render(mode, close)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " x "
            elif position == (3,7):
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
    metadata = {'render.modes': ['human']}

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
            reward = np.random.normal(-0.1,1)
            done = True
        elif self.state == 0:
            if action == 0:
                reward = 0
                done = True
            elif action == 1:
                reward = 0
                self.state = 1
                done = False
            else: print('ERROR')
        info = {}
        state=self.state
        return state, reward, done, info

    def reset(self):
        self.state = 0
        return self.state
  
    def render(self, mode='human'):
        pass

    def close(self):
        pass

class CliffWalkingEnv(gym.Env):
    ''' Cliff Walking Environment
        See the README.md file from https://github.com/caburu/gym-cliffwalking
    '''
    # There is no renderization yet
    # metadata = {'render.modes': ['human']}

    def observation(self, state):
        return state[0] * self.cols + state[1]

    def __init__(self):
        self.nS = 48
        self.nA = 4
        self.rows = 4
        self.cols = 12
        self.start = [0,0]
        self.goal = [0,11]
        self.current_state = None

        # There are four actions: up, down, left and right
        self.action_space = spaces.Discrete(4)

         # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Discrete(self.rows*self.cols)


    def step(self, action):
        new_state = deepcopy(self.current_state)

        if action == 1: #right
            new_state[1] = min(new_state[1]+1, self.cols-1)
        elif action == 2: #down
            new_state[0] = max(new_state[0]-1, 0)
        elif action == 3: #left
            new_state[1] = max(new_state[1]-1, 0)
        elif action == 0: #up
            new_state[0] = min(new_state[0]+1, self.rows-1)
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

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_state_color(self,state_number):
        if state_number in [1,2,3,4,5,6,7,8,9,10]:
            return "grey"
        elif state_number == 11:
            return "green"
        elif state_number == 0:
            return "blue"
        else:
            return "white"