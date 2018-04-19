import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import spaces

import mujoco_py

class PusherDiscreteEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pusher.xml', 5)

        bounds = 0.05*self.model.actuator_ctrlrange.copy()
        num_joints = bounds.shape[0]
        action_list = []
        for i in range(bounds.shape[0]): 
            for j in range(2):
                act = np.zeros((num_joints,))
                act[j] = bounds[i, j]
                action_list.append(act)

        action_list.append(np.zeros((num_joints,)))
        self.idx_to_continuous_action = action_list
        self.action_space = spaces.Discrete(len(self.idx_to_continuous_action))

    def step(self, a):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        #reward_ctrl = - np.square(a).sum()
        reward_ctrl = 0.0
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        continuous_action = self.idx_to_continuous_action[a]
        self.do_simulation(continuous_action, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        while True:
            self.cylinder_pos = np.concatenate([
                    self.np_random.uniform(low=-0.3, high=0, size=1),
                    self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])

    def render(self, mode='human'):
        if mode == 'rgb_array':
            #self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self.sim.render(height, width, camera_name='human')

            #data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        else:
            NotImplementedError()


