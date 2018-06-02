import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import gc
import glob
import os
from random import shuffle
from natsort import natsorted


class MultiReacherEnvV2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        gc.enable()
        utils.EzPickle.__init__(self)

        self.object_xml_paths = natsorted(glob.glob(os.path.join(os.path.dirname(__file__), "assets/testing3/*")))
        self.object_xml_iter = iter(self.object_xml_paths)
        
        self.xml_paths = natsorted(glob.glob(self.object_xml_iter.__next__() + "/*"))
        self.xml_iter = iter(self.xml_paths)

        mujoco_env.MujocoEnv.__init__(self, self.xml_iter.__next__(), 5)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec[:2])

        vec_1 = self.get_body_com("fingertip")-self.get_body_com("cube_0")
        reward_dist_1 = - np.linalg.norm(vec_1[:2])
        
        vec_2 = self.get_body_com("fingertip") - self.get_body_com("cube_1")
        reward_dist_2 = - np.linalg.norm(vec_2)
        #
        reward_dist_tip = - np.linalg.norm(self.get_body_com("fingertip"))

        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_dist_1=reward_dist_1,
                                      reward_dist_2=reward_dist_2,
                                      reward_dist_tip=reward_dist_tip,
                                      reward_ctrl=reward_ctrl)

    def reset_model(self):
        print("resetted")
        qpos = self.np_random.uniform(low=-0.2, high=0.2, size=self.model.nq) + self.init_qpos
        self.goal = np.asarray([0, 0])
        # self.goal[0] = self.np_random.uniform(low=-np.pi, high=np.pi)
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def get_image(self, width=64, height=64):
        return self.sim.render(width, height, camera_name="camera")

    def next(self):
        mujoco_env.MujocoEnv.__init__(self, self.xml_iter.__next__(), 5)

    def next_object(self):
        self.xml_paths = natsorted(glob.glob(self.object_xml_iter.__next__() + "/*"))
        self.xml_iter = iter(self.xml_paths)

        mujoco_env.MujocoEnv.__init__(self, self.xml_iter.__next__(), 5)
