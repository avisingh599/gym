import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py

class GraspEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, substeps=50):
        utils.EzPickle.__init__(self)
        self.substeps = substeps # number of intermediate positions to generate
        mujoco_env.MujocoEnv.__init__(self, 'cartgripper_grasp.xml', 1)


    def step(self, a):
        #vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        #vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        # reward_near = - np.linalg.norm(vec_1)
        # reward_dist = - np.linalg.norm(vec_2)
        # reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        reward = 0.0
        #import IPython; IPython.embed()
        #enquire the qpos here, and then do the interpolation thing

        self.do_pos_simulation_with_substeps(a)
        
        ob = self._get_obs()
        done = False

        is_success = False
        # if reward_dist > -0.17:
        #     is_success = True

        return ob, reward, done, dict(is_success=is_success)

    def do_pos_simulation_with_substeps(self, a):
        #import IPython; IPython.embed()
        qpos_curr = self.sim.data.qpos[:self.action_space.shape[0]]
        step_size = (a - qpos_curr) / self.substeps
        for i in range(self.substeps):
            self.sim.data.ctrl[:] = qpos_curr + (i+1)*step_size
            self.sim.step()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

        # self.goal_pos = np.asarray([0, 0])
        # while True:
        #     self.cylinder_pos = np.concatenate([
        #             self.np_random.uniform(low=-0.3, high=0, size=1),
        #             self.np_random.uniform(low=-0.2, high=0.2, size=1)])
        #     if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
        #         break

        # qpos[-4:-2] = self.cylinder_pos
        # qpos[-2:] = self.goal_pos

        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
        ])


if __name__ == "__main__": 
    env = GraspEnv()
    
    #test drive the env    
    import matplotlib.pyplot as plt

    actions = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.00],
                        [0.0, 0.0, -0.2, 0.0, 0.00],
                        [0.0, 0.0, -0.2, 0.0, 0.06],
                        [0.0, 0.0, 0.1,  0.0, 0.06],
                        [0.0, 0.0, 0.1,  3.0, 0.06],
                        ])
    
    for i in range(actions.shape[0]):
        act = actions[i]
        env.step(act)
        plt.figure(1); plt.clf()
        bla = env.sim.render(200, 200, camera_name='maincam')
        plt.imshow(bla)
        plt.pause(0.5)
