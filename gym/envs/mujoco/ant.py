import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + \
            self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


'''
Difference to Ant-v2:
- Reward
- Initial pos and vel distribution
- Add goal on unit circle
- Observation to contain goal relative position
'''


class AntGoalEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        self.goal_pos = np.array([0.0, 0.0])
        self.init_random_range = 0.5
        # Size of ANT is around 0.56 in diameter
        self.goal_distance = kwargs['goal_distance']
        self.forward_reward_scale = kwargs['forward_reward_scale']

        if kwargs is None or ("final_poses" not in kwargs):
            self.init_with_end = False
        else:
            #print("Setup init with end")
            self.init_with_end = True
            self.init_qpos_pool = kwargs['final_poses']
            self.init_qvel_pool = kwargs['final_vels']
            self.pool_size = len(self.init_qpos_pool)

        if 'set_goal' in kwargs:
            self.set_goal_theta = kwargs['set_goal']

        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.get_body_com("torso")[0:2]
        dist_before = np.linalg.norm(self.goal_pos - posbefore)
        self.do_simulation(a, self.frame_skip)
        posafter = self.get_body_com("torso")[0:2]
        dist_after = np.linalg.norm(self.goal_pos - posafter)

        forward_reward = self.forward_reward_scale * \
            (dist_before - dist_after) / self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.26 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        goal_rel_pos = self.goal_pos - self.sim.data.qpos.flat[0:2]
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            goal_rel_pos,
        ])

    def reset_model(self):
        # Reset init pos and vel
        if self.init_with_end:
            choice = self.np_random.randint(self.pool_size)
            qpos = np.zeros(self.model.nq)
            qpos[2:] = self.init_qpos_pool[choice]
            qvel = self.init_qvel_pool[choice]

        else:
            qpos = self.init_qpos + self.np_random.uniform(
                size=self.model.nq, low=-self.init_random_range, high=self.init_random_range)
            qvel = self.init_qvel + \
                self.np_random.randn(self.model.nv) * self.init_random_range
            qpos[0] = 0
            qpos[1] = 0

        self.set_state(qpos, qvel)

        # Sample goal
        if hasattr(self, 'set_goal_theta'):
            theta = self.set_goal_theta
        else:
            theta = self.np_random.uniform(low=0.0, high=np.pi * 2)
        goal_x = np.cos(theta) * self.goal_distance
        goal_y = np.sin(theta) * self.goal_distance
        self.goal_pos = np.array([goal_x, goal_y])

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
