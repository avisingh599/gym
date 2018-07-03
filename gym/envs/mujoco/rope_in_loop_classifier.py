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

from baselines.classifier.model import Classifier

import imageio
import time
import os
import argparse

data_folder_name = 'rope_data/data/data_individual_tasks/'
if os.environ.get('NVIDIA_DOCKER') is not None:
    IMAGES_DIR = os.path.join('/root/code/', data_folder_name)
else:
    IMAGES_DIR = os.path.join('/media/avi/data/Work/proj_3/openai-baselines/', data_folder_name)

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

class RopeClassifierInLoopEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 reward_mode='thresh',
                 task_id=0,
                 obs_mode='full_state',
                 texture=True,
                 num_success=1000,
                 num_val_success=500,
                 success_thresh=0.5,
                 double_frame_check=False,
                 num_beads=12,
                 init_pos=[0.0, -0.3, 0.0],
                 substeps=50, 
                 log_video=False, 
                 video_substeps = 5, 
                 video_h=500, 
                 video_w=500,
                 camera_name='overheadcam',
                 action_penalty_const=0.0,
                 pixel_distance=False):
        utils.EzPickle.__init__(self)

        #sim params
        self.substeps = substeps # number of intermediate positions to generate

        #env params
        self.num_beads = num_beads
        self.init_pos = init_pos
        self.width = 128
        self.height = 128
        self.texture = texture
        self.obs_mode = obs_mode

        self.pixel_distance = pixel_distance

        #how much data to use
        self.num_success = num_success
        self.num_val_success = num_val_success

        #reward params
        self.action_penalty_const = action_penalty_const
        self.success_thresh = success_thresh
        self.double_frame_check = double_frame_check
        self.reward_mode = reward_mode

        #video params
        self.log_video = log_video
        self.video_substeps = video_substeps
        self.video_h = video_h
        self.video_w = video_w
        self.camera_name = camera_name

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        
        self.sess = tf.get_default_session()

        i_dim = [128, 128, 3]
        recon_dim = [32, 32, 3]

        arch = 'cnn'
        n_filters = [64, 32, 16]
        strides = [2, 1, 1]
        decoder_layers = [200, 100, 50]

        # self.ae_model = AutoEncoderNetwork(arch, i_dim, recon_dim, n_filters, strides, decoder_layers)
        # self.ae_model.load_wt(self.sess, AE_MODEL_PATH)

        arch = 'cnn'
        n_filters = [64, 32, 16]
        strides = [2, 1, 1]
        fc_layers = [200, 100, 50]
        y_dim = 2
        self.cls_model = Classifier(arch, i_dim, y_dim, n_filters, strides, fc_layers)
        learning_rate = 1e-3
        self.cls_model_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.cls_model.loss)
        #self.sess.run(tf.global_variables_initializer())

        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                self.sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        self.sess.run(init_new_vars_op)

        model = rope(num_beads=self.num_beads, 
                    init_pos=self.init_pos,
                    texture=self.texture)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

        #load the reference qpos
        task_dir = os.path.join(IMAGES_DIR, 'task_{}'.format(task_id))
        self.qpos_ref = np.loadtxt(os.path.join(task_dir, 'qpos_original.txt'))

        num_success = self.num_success
        successes = ['success_{}.png'.format(i) for i in range(num_success)]
        success_frames = []
        for k in range(num_success):
            frame = imageio.imread(os.path.join(task_dir, successes[k]))[:, :, :3] / 255.0
            success_frames.append(frame)
        self.success_frames = np.asarray(success_frames)

        #load some validation data too
        num_val_success = self.num_val_success
        successes = ['success_{}.png'.format(i) for i in range(num_success, num_success + num_val_success)]
        success_frames = []
        for k in range(num_val_success):
            frame = imageio.imread(os.path.join(task_dir, successes[k]))[:, :, :3] / 255.0
            success_frames.append(frame)
        self.val_success_frames = np.asarray(success_frames)

        low = np.asarray(4*[-0.4])
        high = np.asarray(4*[0.4])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.last_reward = False
        #when using xmls        
        #mujoco_env.MujocoEnv.__init__(self, 'rope.xml', 5)

    def step(self, a):

        video_frames = self.push(a)
        img = self.sim.render(self.width, self.height, camera_name="overheadcam")/255.0
        prediction = self.cls_model.forward(self.sess, img)

        movement_1 = -1.0*np.linalg.norm(self.sim.data.qpos[:2] - a[:2])
        movement_2 =  -1.0*np.linalg.norm(a[:2] - a[2:])
        action_penalty = movement_1 + movement_2

        #else:        
        if self.double_frame_check:
            if prediction[1] > self.success_thresh:
                if self.last_reward:
                    reward = 1.0
                    is_success_cls = True
                else:
                    reward = 0.0
                    self.last_reward = True
                    is_success_cls = False
            else:
                reward = 0.0
                self.last_reward = False
                is_success_cls = False
        else:
            if prediction[1] > self.success_thresh:
                reward = 1.0
                is_success_cls = True
            else:
                reward = 0.0
                is_success_cls = False

        if self.reward_mode == 'logits':
            reward = np.log(prediction[1])
        elif self.reward_mode == 'probs':
            reward = prediction[1]
        elif self.reward_mode == 'thresh':
            pass
        elif self.reward_mode == 'pixel_distance':
            pixel_space_distances = []
            for i in range(5):
                pixel_space_distances.append(np.linalg.norm(self.success_frames[i] - img))
            pixel_dist = np.min(np.asarray(pixel_space_distances))
            reward = -1.0*pixel_dist
        else:
            raise NotImplementedError

        reward += action_penalty*self.action_penalty_const
        ob = self._get_obs()
        done = False

        #success is determined by classifier as of now
        if prediction[1] > 0.5:
            is_success_borderline = True
        else:
            is_success_borderline = False

        distance = calculate_distance(self.qpos_ref, self.sim.data.qpos, self.num_beads)

        return ob, reward, done, dict(is_success_cls=is_success_cls,
                                      is_success_borderline=is_success_borderline, 
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
        if self.obs_mode == 'full_state':
            obs = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
                ])

        elif self.obs_mode == 'ae_feats':
            img = self.sim.render(self.width, self.height, camera_name="overheadcam")/1.0

            feed_dict = {
                self.ae_model.i : np.expand_dims(img, axis=0),
                self.ae_model.is_train: False,
            }

            feats =  self.sess.run(self.ae_model.feats, feed_dict=feed_dict)[0] 

            obs = np.concatenate([
            feats,
            self.sim.data.qpos.flat[:6],
            self.sim.data.qvel.flat[:6],
            ])

        else:
            raise NotImplementedError()

        return obs

if __name__ == "__main__":
    with tf.Session() as sess:
        env = RopeClassifierInLoopEnv(obs_mode='full_state')
        env.push(np.zeros(4,))
        bla = env.step(np.zeros(4,))
        print(bla[0].shape)
    #img = env.sim.render(env.width, env.height, camera_name="overheadcam")/255.0
