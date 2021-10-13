from gym_minigrid.register import register
from gym_minigrid.envs.empty import EmptyEnv
import numpy as np
import hashlib


def hash_obs(observation, size=16):
    """Compute a hash that uniquely identifies the current state of the environment.
    :param size: Size of the hashing
    """
    sample_hash = hashlib.sha256()

    img = observation['image']
    agent_dir = observation['direction']
    to_encode = [img.tolist(), agent_dir]
    for item in to_encode:
        sample_hash.update(str(item).encode('utf8'))

    return sample_hash.hexdigest()[:size]


class EmptyEnvDenseReward_StochRew(EmptyEnv):
    def __init__(self,
                 size=8,
                 agent_start_pos=(1, 1),
                 agent_start_dir=0):
        super().__init__(size, agent_start_pos, agent_start_dir)
        self.nS = self.observation_space.spaces.__sizeof__()
        self.nA = 3
        self.obs_idx = dict()

    def step(self, action):
        # TODO: add hashing here
        self.step_count += 1

        reward = np.random.choice([-12,10])
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            # reward=0.
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            # reward=0.
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            #reward = -.1
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                #reward = self._reward()
                reward = 10
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        # hash the observation and map to unique index
        obs = hash_obs(obs)
        if obs not in self.obs_idx:
            self.obs_idx[obs] = len(self.obs_idx)

        return self.obs_idx[obs], reward, done, {}

    def custom_reset(self):

        obs = self.reset()

        obs = hash_obs(obs)
        if obs not in self.obs_idx:
            self.obs_idx[obs] = len(self.obs_idx)

        return self.obs_idx[obs]


class EmptyEnvDense5x5StochRew(EmptyEnvDenseReward_StochRew):
    def __init__(self, **kwargs):
        super().__init__(size=5, **kwargs)
        #print('has obs idx?', self.obs_idx)


register(
    id='MiniGrid-EmptyDense-5x5StochRew-v0',
    entry_point='env_dense_stoch_rewards:EmptyEnvDense5x5StochRew'
)
