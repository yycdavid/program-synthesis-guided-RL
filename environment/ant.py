import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

'''
- One reset will accept a starting position. Called at the start of each rollout
- Second reset will resample a pose at the same (x, y) position. Called when the ANT flips.
- Each craft scenario should have a AntCraftEnv
- A set goal function
'''


class AntCraftEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, start_states, blocks, pos_ant):
        self.goal_pos = np.array([0.0, 0.0])
        self.init_random_range = 0.0

        self.init_qpos_pool = start_states['final_poses']
        self.pool_size = len(self.init_qpos_pool)

        self.init_blocks = blocks
        self.current_pos = pos_ant
        self.current_blocks = self.init_blocks.copy()

        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        #mujoco_env.MujocoEnv.__init__(self, 'ant_craft.xml', 5)
        utils.EzPickle.__init__(self, start_states, blocks, pos_ant)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        state = self.state_vector()
        if self._in_block(state[:2]):
            # Return to previous location if ant hits a block
            qpos = self.sim.data.qpos.flat.copy()
            qvel = self.sim.data.qvel.flat.copy()
            qpos[0] = self.current_pos[0]
            qpos[1] = self.current_pos[1]
            self.set_state(qpos, qvel)

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.26 and state[2] <= 1.0
        done = not notdone
        obs = self.get_obs()
        self.current_pos = self.sim.data.qpos.flat[:2].copy()
        return obs, None, done, None

    def _in_block(self, pos):
        in_block = False
        for (craft_pos, block_range) in self.current_blocks.items():
            x = pos[0]
            y = pos[1]
            x_min, x_max, y_min, y_max = block_range
            if (x >= x_min) and (x <= x_max) and (y >= y_min) and (y <= y_max):
                in_block = True
                break
        return in_block

    def remove_block(self, craft_pos):
        del self.current_blocks[craft_pos]

    def get_obs(self):
        goal_rel_pos = self.goal_pos - self.sim.data.qpos.flat[0:2]
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            goal_rel_pos,
        ])

    def get_pos(self):
        return self.current_pos

    def set_goal(self, goal_pos):
        self.goal_pos = goal_pos

    def reset_rollout(self, start_pos):
        # Reset init pos and vel
        self._reset_at_pos(start_pos)
        self.current_pos = start_pos
        self.current_blocks = self.init_blocks.copy()

    def reset_inplace(self):
        self._reset_at_pos(self.current_pos)

    def reset_model(self):
        pass

    def _reset_at_pos(self, pos):
        choice = self.np_random.randint(self.pool_size)
        qpos = np.zeros(self.model.nq)
        qpos[2:] = self.init_qpos_pool[choice]
        qpos[0] = pos[0]
        qpos[1] = pos[1]
        qvel = np.zeros(self.model.nv)

        self.set_state(qpos, qvel)

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
