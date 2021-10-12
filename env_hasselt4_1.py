import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

# same as BasicEnv, with one difference: the reward for each action is a normal variable
# purpose is to see if we can use libraries

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