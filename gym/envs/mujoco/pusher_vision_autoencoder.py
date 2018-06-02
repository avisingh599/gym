import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py

import sys
sys.path.insert(0,'../baselines/baselines/autoencoder')
from train import AutoEncoderNetwork
import tensorflow as tf
import cv2

class PusherVisionAutoEncoderEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)

        #cnn params
        arch = 'cnn'
        n_filters = [64, 32, 16]
        strides = [2, 1, 1]
        i_dim = [64, 64, 3]
        recon_dim = [32, 32, 3]
        decoder_layers = [200, 100, 50]

        #chkpt_file = '../baselines/ae_checkpoints/2018-05-29_15-57-56_mjc150_dropout_bn_cnn_filters_64_32_16_/weights_050000.pkl'
        chkpt_file = '../baselines/ae_checkpoints/2018-05-29_19-01-16_cv2_mjc150_dropout_bn_cnn_filters_64_32_16_/weights_050000.pkl'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.img_model = AutoEncoderNetwork(arch, i_dim, recon_dim, n_filters, strides, decoder_layers)
        self.img_model.load_wt(self.sess, chkpt_file)

        mujoco_env.MujocoEnv.__init__(self, 'pusher.xml', 5)

    def step(self, a):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        #reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        reward = reward_dist + 0.0 * reward_ctrl + 0.2 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        is_success = False
        if reward_dist > -0.17:
            is_success = True
        return ob, reward, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl, is_success=is_success)

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
        #rgb = self.sim.render(64, 64, camera_name='camera')
        rgb = self.sim.render(256, 256, camera_name='camera')
        #import IPython; IPython.embed()
        rgb = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
        rgb = rgb/255.0

        feed_dict = {
            self.img_model.i : np.expand_dims(rgb, axis=0),
            self.img_model.is_train: False,
        }
        
        img_feats = self.sess.run(self.img_model.feats, feed_dict=feed_dict) 

        return np.concatenate([
            img_feats[0],
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("goal"),
        ])

if __name__ == "__main__":
    env = PusherVisionAutoEncoderEnv()
    ob = env.reset()
