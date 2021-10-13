from gym_minigrid.minigrid import *
from gym_minigrid.envs.distshift import DistShiftEnv
from gym_minigrid.register import register
import numpy as np
from hash_obs import *

class LavaDetEnv(DistShiftEnv):
    """
    Deterministic environment with lava.
    """

    def __init__(
        self,
        width=9,
        height=7,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        strip2_row=5
    ):
        super().__init__(
            width,
            height,
            agent_start_pos,
            agent_start_dir,
            strip2_row
        )
        self.nS = self.observation_space.spaces.__sizeof__()
        self.nA = 3
        self.obs_idx = dict()

    def step(self, action):

        self.step_count += 1

        # Default is reward of -1 per step, and no termination; may be adjusted based on action and position
        reward = -1.0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True
                reward = -10.0

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

class LavaDetEnv9x7(LavaDetEnv):
    def __init__(self, **kwargs):
        super().__init__(width=9, height=7, **kwargs)

register(
    id='MiniGrid-LavaDet-9x7-v0',
    entry_point='env_lava_det:LavaDetEnv9x7'
)

