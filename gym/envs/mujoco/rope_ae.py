import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import error, spaces

from gym.envs.mujoco.dynamic_mjc.rope import rope

import mujoco_py

import tensorflow as tf
from tensorflow.python.platform import flags
from visual_mpc.one_shot_predictor import OneShotPredictor
from train import AutoEncoderNetwork

import imageio
import time
import os
import argparse

model_name = 'cnn_ae_weights_070000.pkl'

if os.environ.get('NVIDIA_DOCKER') is not None:
    AE_MODEL_PATH = '/dsae/models/{}'.format(model_name)
    IMAGES_DIR = '/rope_data/data/data_rand_act_1/'
else:
    AE_MODEL_PATH = '/media/avi/data/Work/proj_3/openai-baselines/dsae/models/{}'.format(model_name)
    IMAGES_DIR = '/media/avi/data/Work/proj_3/openai-baselines/rope_data/data/data_rand_act_1/'

def get_beads_xy(qpos, num_beads):
    init_joint_offset = 6
    num_free_joints = 7

    xy_list = []
    for j in range(num_beads):
        offset = init_joint_offset + j*num_free_joints
        xy_list.append(qpos[offset:offset+2])

    return np.asarray(xy_list)

def get_com(xy_list):
    return np.mean(xy_list, axis=0)

def calculate_distance(qpos1, qpos2, num_beads):

    xy1 = get_beads_xy(qpos1, num_beads)
    xy2 = get_beads_xy(qpos2, num_beads)

    com1 = get_com(xy1)
    com2 = get_com(xy2)

    xy1_translate = xy1 + (com2 - com1)

    distance = np.linalg.norm(xy1_translate - xy2)

    return distance

class RopeAEEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 task_id=0,
                 texture=True,
                 success_thresh=0.05,
                 num_beads=12,
                 init_pos=[0.0, -0.3, 0.0],
                 substeps=50, 
                 log_video=False, 
                 video_substeps = 5, 
                 video_h=500, 
                 video_w=500,
                 camera_name='overheadcam',
                 action_penalty_const=0.0,):
        utils.EzPickle.__init__(self)

        #sim params
        self.substeps = substeps # number of intermediate positions to generate

        #env params
        self.num_beads = num_beads
        self.init_pos = init_pos
        self.width = 128
        self.height = 128
        self.texture = texture

        #reward params
        self.action_penalty_const = action_penalty_const
        self.success_thresh = success_thresh

        #video params
        self.log_video = log_video
        self.video_substeps = video_substeps
        self.video_h = video_h
        self.video_w = video_w
        self.camera_name = camera_name

        model = rope(num_beads=self.num_beads, 
                    init_pos=self.init_pos,
                    texture=self.texture)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)
        
        #load meta classifier model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        i_dim = [128, 128, 3]
        recon_dim = [32, 32, 3]

        arch = 'cnn'
        n_filters = [64, 32, 16]
        strides = [2, 1, 1]
        decoder_layers = [200, 100, 50]

        self.ae_model = AutoEncoderNetwork(arch, i_dim, recon_dim, n_filters, strides, decoder_layers)
        self.ae_model.load_wt(self.sess, AE_MODEL_PATH)

        #OneShotPredictor(METACLASSIFIER_MODEL_PATH, self.sess)

        #load example success images
        task_dir = '{}/task_{}/'.format(IMAGES_DIR, task_id)
        successes = ['success_0.png', 'success_1.png', 'success_2.png', 'success_3.png', 'success_4.png']
        success_frames = []

        for k in range(5):
            frame = imageio.imread(task_dir + successes[k])[:, :, :3] / 1.0
            success_frames.append(frame)

        self.success_frames = np.asarray(success_frames)

        feed_dict = {
            self.ae_model.i : self.success_frames,
            self.ae_model.is_train: False,
        }
        self.success_feats = self.sess.run(self.ae_model.feats, feed_dict=feed_dict) 

        # self.success_pred.backward(self.success_frames)
        # self.success_pred.construct_model()

        #load the reference qpos
        self.qpos_ref = np.loadtxt(os.path.join(task_dir, 'qpos_original.txt'))

        low = np.asarray(4*[-0.4])
        high = np.asarray(4*[0.4])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.last_reward = False
        #when using xmls        
        #mujoco_env.MujocoEnv.__init__(self, 'rope.xml', 5)

    def step(self, a):

        video_frames = self.push(a)
        #vae takes in images which are in 0 to 255.0 range
        img = self.sim.render(self.width, self.height, camera_name="overheadcam")/1.0
        
        #prediction = self.ae_model.forward(img)
        feed_dict = {
            self.ae_model.i : np.expand_dims(img, axis=0),
            self.ae_model.is_train: False,
        }
        
        img_feats =  self.sess.run(self.ae_model.feats, feed_dict=feed_dict) 

        ae_space_distances = []
        for i in range(self.success_feats.shape[0]):
            ae_space_distances.append(np.linalg.norm(self.success_feats[i] - img_feats))

        #make the min distance from the features the reward
        vae_dist = np.min(np.asarray(ae_space_distances))
        reward = -1.0*vae_dist

        movement_1 = -1.0*np.linalg.norm(self.sim.data.qpos[:2] - a[:2])
        movement_2 =  -1.0*np.linalg.norm(a[:2] - a[2:])
        action_penalty = movement_1 + movement_2

        distance = calculate_distance(self.qpos_ref, self.sim.data.qpos, self.num_beads)
        if distance < self.success_thresh:
            is_success = True
        else:
            is_success = False

        reward += action_penalty*self.action_penalty_const

        ob = self._get_obs()
        done = False
        distance = calculate_distance(self.qpos_ref, self.sim.data.qpos, self.num_beads)

        return ob, reward, done, dict(is_success=is_success,
                                      vae_dist=vae_dist,
                                      video_frames=video_frames,
                                      action_penalty=action_penalty,
                                      prediction_img=img,
                                      oracle_distance=distance)


    def pick_place(self, a):
        x_start = a[0]
        y_start = a[1]
        x_end = a[2]
        y_end = a[3]

        z_min = -0.1
        z_max = +0.05

        torque_max = +10.0
        torque_neutral = 0.0
        torque_min = -1*torque_max

        actions = np.asarray(
               [[0.0, -0.2, z_max, 0.0, torque_neutral], #neutral position
                [x_start, y_start, z_max, 0.0, torque_neutral], #get close
                [x_start, y_start, z_min, 0.0, torque_neutral], #go down
                [x_start, y_start, z_min, 0.0, torque_max], #grasp
                [x_start, x_start, z_max,  0.0, torque_max], #go up
                [x_end,y_end,z_max,0.0,torque_max], #move
                [x_end,y_end, z_min, 0.0, torque_max], #go down
                [x_end,y_end, z_min, 0.0, torque_neutral], #drop 
                [x_end,y_end, z_max, 0.0, torque_neutral], #go back up 
                [0.0, -0.2, z_max, 0.0, torque_min],#neutral position, open gripper
                ])

        video_frames = []
        for i in range(actions.shape[0]):
            video_frames.append(self.do_pos_simulation_with_substeps(actions[i]))

        return video_frames

    def push(self, a):
        x_start = a[0]
        y_start = a[1]
        x_end = a[2]
        y_end = a[3]

        x_neutral = -0.5
        y_neutral = -0.5
        z_min = -0.1
        z_max = +0.05
        torque_max = +10.0
        torque_neutral = 0.0
        torque_min = -1*torque_max

        actions = np.asarray(
               [
                # [x_neutral, y_neutral, z_max, 0.0, torque_max], #neutral position
                [x_start, y_start, z_max, 0.0, torque_max], #get close, close gripper
                [x_start, y_start, z_min, 0.0, torque_neutral], #go down
                [x_end,y_end,z_min,0.0,torque_neutral], #move
                [x_end,y_end, z_max, 0.0, torque_max], #go back up, close gripper 
                [x_neutral, y_neutral, z_max, 0.0, torque_max],#neutral position, open gripper
                [x_neutral, y_neutral, z_max, 0.0, torque_max],#neutral position, open gripper
                ])

        video_frames = []
        for i in range(actions.shape[0]):
            video_frames.append(self.do_pos_simulation_with_substeps(actions[i]))

        return video_frames

    def do_pos_simulation_with_substeps(self, a):
        qpos_curr = self.sim.data.qpos[:self.action_space.shape[0]]
        a_pos = a[:self.action_space.shape[0]]

        step_size = (a_pos - qpos_curr) / self.substeps

        if self.log_video:
            video_frames = np.zeros((int(self.substeps/self.video_substeps), self.video_h, self.video_w, 3))
        else:
            video_frames = None
        for i in range(self.substeps):
            self.sim.data.ctrl[:-1] = qpos_curr + (i+1)*step_size
            #torque control on the gripper
            self.sim.data.ctrl[-1] = a[-1]
            self.sim.step()
            if i%self.video_substeps == 0 and self.log_video :
                video_frames[int(i/self.video_substeps)] = self.sim.render(self.video_h, self.video_w, camera_name=self.camera_name)

        return video_frames

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel 
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

if __name__ == "__main__":
    env = RopeAEEnv()
    env.push(np.zeros(4,))
    env.step(np.zeros(4,))

    img = env.sim.render(env.width, env.height, camera_name="overheadcam")/255.0
