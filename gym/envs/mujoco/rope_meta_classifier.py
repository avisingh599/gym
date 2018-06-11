import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import error, spaces

from gym.envs.mujoco.dynamic_mjc.rope import rope

import mujoco_py

import tensorflow as tf
from tensorflow.python.platform import flags
from visual_mpc.one_shot_predictor import OneShotPredictor
import imageio
import time

import argparse
tf.flags._global_parser = argparse.ArgumentParser() #hacky stuff to clear the flags
FLAGS = flags.FLAGS

#temp flags for the OneShotModel to work, remove them later

flags.DEFINE_integer('T', 20, '# of time steps in trajectory.')
flags.DEFINE_integer('M', 200, '# of action sequences to sample.')
flags.DEFINE_integer('H', 10, 'length of planning horizon.')
flags.DEFINE_integer('K', 8, '# of action sequences to refit Gaussian distribution with.')
flags.DEFINE_integer('repeat', 5, '# of times steps to repeat actions.')
flags.DEFINE_float('initial_std', 0.0125, 'initial standard deviation of Gaussian distribution.')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 40000, 'number of metatraining iterations.')
flags.DEFINE_integer('meta_batch_size', 10, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1.0, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_bool('grad_clip', True, 'use gradient clipping')
flags.DEFINE_float('clip_min', -80.0, 'minimum for gradient clipping')
flags.DEFINE_float('clip_max', 80.0, 'maximum for gradient clipping')
flags.DEFINE_bool('stop_grad', True, 'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_integer('num_tasks', 150, 'number of tasks in dataset')
flags.DEFINE_integer('num_examples', 30, 'number of positive examples per task to use for training')
flags.DEFINE_integer('im_height', 128, 'height of input image')
flags.DEFINE_integer('im_width', 128, 'width of input image')
flags.DEFINE_integer('im_channels', 3, 'number of channels in input image')

## Model options
flags.DEFINE_string('norm', 'layer_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_conv_layers', 3, 'number of convolutional layers')
flags.DEFINE_integer('num_filters', 16, 'number of filters for conv nets.')
flags.DEFINE_integer('num_fc_layers', 3, 'number of fully connected layers')
flags.DEFINE_integer('hidden_dim', 50, 'hidden dimension of fully connected layers')
flags.DEFINE_bool('fc_bt', False, 'use bias transformation for the first fc layer')
flags.DEFINE_integer('bt_dim', 0, 'the dimension of bias transformation for FC layers')
flags.DEFINE_bool('fp', False, 'use feature spatial soft-argmax')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', 5, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step during training. (use if you want to test with a different value)') # 0.1 for omniglot

## Debugging options
flags.DEFINE_bool('debug', False, 'debugging mode')

flags.DEFINE_bool('resnet_feats', False, 'resnet_feats')
flags.DEFINE_bool('vgg_path', False, 'resnet_feats')


class RopeMetaClassifierEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 task_id=0,
                 num_beads=12,
                 init_pos=[0.0, -0.3, 0.0],
                 substeps=50, 
                 log_video=False, 
                 video_substeps = 5, 
                 video_h=500, 
                 video_w=500,
                 camera_name='overheadcam',
                 action_penalty_const=0,):
        utils.EzPickle.__init__(self)
        
        #sim params
        self.substeps = substeps # number of intermediate positions to generate
        
        #env params
        self.num_beads = num_beads
        self.init_pos = init_pos
        self.width = 128
        self.height = 128
        #reward params
        self.action_penalty_const = action_penalty_const

        #video params
        self.log_video = log_video
        self.video_substeps = video_substeps
        self.video_h = video_h
        self.video_w = video_w
        self.camera_name = camera_name

        model = rope(num_beads=self.num_beads, init_pos=self.init_pos)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)
        
        #load meta classifier model
        pretrained_model_path = '/media/avi/data/Work/proj_3/rope_models/rope_model_0/model99999'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.success_pred = OneShotPredictor(pretrained_model_path, self.sess)

        #load example success images
        task_dir = '/media/avi/data/Work/proj_3/rope_data/data/task_{}/'.format(task_id)
        successes = ['success_0.png', 'success_1.png', 'success_2.png', 'success_3.png', 'success_4.png']
        success_frames = []

        for k in range(5):
            frame = imageio.imread(task_dir + successes[k])[:, :, :3] / 255.0
            success_frames.append(frame)

        self.success_frames = np.asarray(success_frames)
        self.success_pred.backward(self.success_frames)
        self.success_pred.construct_model()


        low = np.asarray(4*[-0.4])
        high = np.asarray(4*[0.4])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        #when using xmls        
        #mujoco_env.MujocoEnv.__init__(self, 'rope.xml', 5)

    def step(self, a):


        video_frames = self.push(a)

        img = self.sim.render(self.width, self.height, camera_name="overheadcam")/255.0
        prediction = self.success_pred.forward(img)

        movement_1 = -1.0*np.linalg.norm(self.sim.data.qpos[:2] - a[:2])
        movement_2 =  -1.0*np.linalg.norm(a[:2] - a[2:])
        action_penalty = movement_1 + movement_2

        if prediction[0,1] > 0.85:
            if self.last_reward:
                reward = 1.0
            else:
                reward = 0.0
                self.last_reward = True
        else:
            reward = 0.0
            self.last_reward = False

        reward += action_penalty*self.action_penalty_const

        ob = self._get_obs()
        done = False

        #success is determined by classifier as of now
        if prediction[0,1] > 0.5:
            is_success_cls = True
        else:
            is_success_cls = False

        return ob, reward, done, dict(is_success_cls=is_success_cls, 
                                      video_frames=video_frames,
                                      action_penalty=action_penalty)


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

        x_neutral = 0.0
        y_neutral = -0.2
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
                # [x_neutral, y_neutral, z_max, 0.0, torque_max],#neutral position, open gripper
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
   pass