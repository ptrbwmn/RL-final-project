import gym_minigrid
from gym_minigrid.wrappers import *
import numpy as np
from collections import defaultdict
import sys
import random
import time
from policy import EpsilonGreedyPolicy, EpsilonGreedyPolicy_Double_Q
from q_learning import sarsa, q_learning, double_q_learning
#from plotting import plot
from tqdm import tqdm as _tqdm
import numpy as np
import random

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer


# derive MiniGridEnv class
#class TestEnv(gym_minigrid.minigrid.MiniGridEnv(gym.Env)):
class RewardWrapper(gym.core.ObservationWrapper):
    def __init__(self,env):
        super().__init__(env)
    #def reward(self,rew):
        #if rew==0:
        #    return -0.1
        #else:
        #return rew
    def step(self, action):
        self.env.step_count += 1

        reward = -0.1
        done = False

        # Get the position in front of the agent
        fwd_pos = self.env.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.env.grid.get(*fwd_pos)

        # Rotate left
        if action == self.env.actions.left:
            #reward=0.
            self.env.agent_dir -= 1
            if self.env.agent_dir < 0:
                self.env.agent_dir += 4

        # Rotate right
        elif action == self.env.actions.right:
            #reward=0.
            self.env.agent_dir = (self.env.agent_dir + 1) % 4

        # Move forward
        elif action == self.env.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.env.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self.env._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.env.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.env.carrying is None:
                    self.env.carrying = fwd_cell
                    self.env.carrying.cur_pos = np.array([-1, -1])
                    self.env.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.env.actions.drop:
            if not fwd_cell and self.env.carrying:
                self.env.grid.set(*fwd_pos, self.env.carrying)
                self.env.carrying.cur_pos = fwd_pos
                self.env.carrying = None

        # Toggle/activate an object
        elif action == self.env.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self.env, fwd_pos)

        # Done action (not used by default)
        elif action == self.env.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.env.step_count >= self.env.max_steps:
            done = True

        obs = self.env.gen_obs()

        return obs, reward, done, {} 

#env = gym.make('MiniGrid-Empty-8x8-v0')
env = RewardWrapper(gym.make('MiniGrid-Empty-5x5-v0'))

obs = env.env.reset() # This now produces an RGB tensor only
env.env.render()
input('press any key to start')
nS = env.env.observation_space.spaces.__sizeof__()
# Q={}
# if env.hash() not in Q:
#     Q[env.hash()] = (0,0,0)
Q=np.zeros((nS,3))
Q=np.ones((nS,3))

def q_learning_(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
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
    HT = {}
    for i_episode in range(num_episodes):#tqdm(range(num_episodes)):
        i = 0
        R = 0
        print('episode',i_episode,'-',end='')
        start_state = env.env.reset()
        if env.env.hash() not in HT:
            HT[env.env.hash()] = len(HT.keys())

        done = False
        while not done:
            start_state = HT[env.env.hash()]
            start_action = policy.sample_action(start_state)
            print(start_action,end='')
            new_step = env.step(start_action)
            new_state = new_step[0]
            if env.env.hash() not in HT:
                HT[env.env.hash()] = len(HT.keys())
            new_state = HT[env.env.hash()]
            reward = new_step[1]
            done = new_step[2]
            Qnew=np.max(policy.Q[new_state])
            if done:
                Qnew=0
            policy.Q[start_state][start_action] = policy.Q[start_state][start_action] + alpha*(reward + \
                                            discount_factor*Qnew - policy.Q[start_state,start_action])
            #print(Q)
            start_state = new_state
            i+=1
            R+=(discount_factor**i)*reward
        
        stats.append((i, R))
        print(', steps:',i,', reward:',R)
    episode_lengths, episode_returns = zip(*stats)
    return [Q], episode_returns, policy, episode_lengths

policy=EpsilonGreedyPolicy(Q,0.1)

Q_q_learning1, episode_returns1, policy_q_learning1, episode_lengths1 = q_learning_(env, policy, Q, 60)
import matplotlib.pyplot as plt
plt.plot(episode_lengths1)
plt.show()
input('press key to end')

