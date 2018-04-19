import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py

class Pusher1DVisionEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pusher.xml', 5)

    def step(self, a):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
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
            #self.cylinder_pos = np.concatenate([
            #        self.np_random.uniform(low=-0.3, high=0, size=1),
            #        self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            # self.cylinder_pos = np.concatenate([
            #         self.np_random.uniform(low=-0.3, high=0, size=1),
            #         np.asarray([0.0])])
            self.cylinder_pos = np.concatenate([
                   np.asarray([-0.15]),
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
        rgb = self.sim.render(64, 64, camera_name='camera')
        rgb_slice = rgb[21, :, :]

        #rgb 18 to 27 or so shows the entire block
        #from scipy.misc import imshow
        #import IPython; IPython.embed()
        
        return np.concatenate([
            rgb_slice.flat[:]/255.0,
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("goal"),
        ])

#if __name__ == "__main__": 
#    env = Pusher1DVisionEnv()
#    img = env.reset_model()
#    img = img[:64*64*3]


