from gym_minigrid.register import register
import matplotlib.pyplot as plt
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.envs.empty import EmptyEnv
import numpy as np
from collections import defaultdict
import sys
import random
import time
from policy import EpsilonGreedyPolicy, EpsilonGreedyPolicy_Double_Q
from q_learning import q_learning, double_q_learning
#from plotting import plot
from tqdm import tqdm as _tqdm
import numpy as np
import random

from environments import LavaWorld

env=LavaWorld()
a=env.reset()

