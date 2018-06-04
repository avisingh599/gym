import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py

class BlockStackEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, substeps=50):
        utils.EzPickle.__init__(self)
        self.substeps = substeps # number of intermediate positions to generate
        mujoco_env.MujocoEnv.__init__(self, 'block_stack.xml', 5)

    def step(self, a):
        #vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        #vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        # reward_near = - np.linalg.norm(vec_1)
        # reward_dist = - np.linalg.norm(vec_2)
        # reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        #import IPython; IPython.embed()
        #enquire the qpos here, and then do the interpolation thing
        #self.do_simulation(a, self.frame_skip)
        
        reward = 0.0
        #TODO maybe define some reward function here at some point
        self.do_simulation_with_substeps(a)
        
        ob = self._get_obs()
        done = False

        is_success = False
        # if reward_dist > -0.17:
        #     is_success = True

        return ob, reward, done, dict(is_success=is_success)

    def do_simulation_with_substeps(self, a):
        qpos_curr = self.sim.data.qpos[:self.action_space.shape[0]]
        step_size = (a - qpos_curr) / self.substeps
        for i in range(self.substeps):
            self.sim.data.ctrl[:] = qpos_curr + (i+1)*step_size
            self.sim.data.ctrl[-1] = a[-1]
            self.sim.step()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos
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
    env = BlockStackEnv()
    
    #test drive the env    
    import matplotlib.pyplot as plt
    time_wait = 1.5
    #torque control on the gripper
    torque_max = 10.0
    actions = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.00],
                    [0.0, 0.0, -0.25, 0.0, 0.00],
                    [0.0, 0.0, -0.25, 0.0, torque_max],
                    [0.0, 0.0, -0.25, 0.0, torque_max],
                    [0.0, 0.0, -0.25, 0.0, torque_max],
                    [0.0, 0.0, 0.1,  0.0, torque_max],
                    [0.1, 0.1, 0.1,  0.0, torque_max],
                    [0.1, 0.1, 0.1,  0.0, 0.0],
                    [0.1, 0.1, 0.1,  0.0, 0.0],
                    ])

    for i in range(actions.shape[0]):
        act = actions[i]
        env.step(act)
        plt.figure(1); plt.clf()
        bla = env.sim.render(200, 200, camera_name='leftcam')
        plt.imshow(bla)
        plt.pause(time_wait)
        print(env.sim.data.qpos[:actions.shape[1]])
