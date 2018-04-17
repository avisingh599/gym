import itertools
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import spaces

class ReacherDiscreteNoVelocityFixedGoalEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, num_bins=5):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)
        
        assert(num_bins >=2)

        bounds = self.model.actuator_ctrlrange.copy()
        action_ranges = [
            np.linspace(bounds[i, 0], bounds[i, 1], num_bins)
            for i in range(bounds.shape[0])
        ]
        self.idx_to_continuous_action = [
            np.array(x) for x in itertools.product(*action_ranges)
        ]

        self.action_space = spaces.Discrete(len(self.idx_to_continuous_action))

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        
        continuous_action = self.idx_to_continuous_action[a]
        self.do_simulation(continuous_action, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        # while True:
        #     self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
        #     if np.linalg.norm(self.goal) < 2:
        #         break

        self.goal = np.asarray([0.1, 0.0])
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
            #self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def render(self, mode='human'):
        if mode == 'rgb_array':
            #self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self.sim.render(height, width, camera_name='camera')

            #data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            width, height = 500, 500
            self.sim.render(height, width, camera_name='camera')
            #self.sim.render()

if __name__ == "__main__": 

    env = ReacherDiscreteNoVelocityFixedGoalEnv(num_bins = 5)

