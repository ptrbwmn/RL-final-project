from gym_minigrid.register import register
from gym_minigrid.envs.empty import EmptyEnv
import numpy as np


class EmptyEnvDenseReward(EmptyEnv):
    def __init__(self,
                 size=8,
                 agent_start_pos=(1, 1),
                 agent_start_dir=0):
        super().__init__(size, agent_start_pos, agent_start_dir)

    def step(self, action):
        # TODO: add hashing here
        self.step_count += 1

        reward = -0.1
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
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
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

        return obs, reward, done, {}


class EmptyEnvDense5x5(EmptyEnvDenseReward):
    def __init__(self, **kwargs):
        super().__init__(size=5, **kwargs)


register(
    id='MiniGrid-EmptyDense-5x5-v0',
    entry_point='env_dense:EmptyEnvDense5x5'
)
