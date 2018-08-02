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


## CORL 2018 submission time parameters
# model_name = 'model9000'
# ae_model_name = 'cnn_ae_weights_070000.pkl'

# if os.environ.get('NVIDIA_DOCKER') is not None:
#     AE_MODEL_PATH = '/root/code/dsae/models/{}'.format(ae_model_name)
#     METACLASSIFIER_MODEL_PATH = '/root/code/rope_models/rope_model_rand_act_1/{}'.format(model_name)
#     IMAGES_DIR = '/root/code/rope_data/data/data_rand_act_1/'
# else:
#     AE_MODEL_PATH = '/media/avi/data/Work/proj_3/openai-baselines/dsae/models/{}'.format(ae_model_name)
#     METACLASSIFIER_MODEL_PATH = '/media/avi/data/Work/proj_3/openai-baselines/rope_models/rope_model_rand_act_1/{}'.format(model_name)
#     IMAGES_DIR = '/media/avi/data/Work/proj_3/openai-baselines/rope_data/data/data_rand_act_1/'

## post submission
model_name = 'model52000'
ae_model_name = 'cnn_ae_weights_070000.pkl'

if os.environ.get('NVIDIA_DOCKER') is not None:
    AE_MODEL_PATH = '/root/code/dsae/models/{}'.format(ae_model_name)
    METACLASSIFIER_MODEL_PATH = '/root/code/rope_models/Aug_02_new_tasks_5_shot_conv_3_16_fc_3_50_color_no_dups/{}'.format(model_name)
    IMAGES_DIR = '/root/code/rope_data/data/rope_val_08_02/data/validation_tasks/'
else:
    AE_MODEL_PATH = '/media/avi/data/Work/proj_3/openai-baselines/dsae/models/{}'.format(ae_model_name)
    METACLASSIFIER_MODEL_PATH = '/media/avi/data/Work/proj_3/openai-baselines/rope_models/Aug_02_new_tasks_5_shot_conv_3_16_fc_3_50_color_no_dups/{}'.format(model_name)
    IMAGES_DIR = '/media/avi/data/Work/proj_3/openai-baselines/rope_data/data/rope_val_08_02/data/validation_tasks/'

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

class RopeMetaClassifierEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 task_id=0,
                 reward_mode='thresh',
                 obs_mode='full_state',
                 texture=True,
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
                 ):
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
        self.reward_mode = reward_mode

        #reward params
        self.action_penalty_const = action_penalty_const
        self.success_thresh = success_thresh
        self.double_frame_check = double_frame_check

        #video params
        self.log_video = log_video
        self.video_substeps = video_substeps
        self.video_h = video_h
        self.video_w = video_w
        self.camera_name = camera_name
        
        #load meta classifier model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.success_pred = OneShotPredictor(METACLASSIFIER_MODEL_PATH, self.sess)

        #load example success images
        task_dir = '{}/task_{}/'.format(IMAGES_DIR, task_id)
        successes = ['success_0.png', 'success_1.png', 'success_2.png', 'success_3.png', 'success_4.png']
        success_frames = []

        for k in range(5):
            frame = imageio.imread(task_dir + successes[k])[:, :, :3] / 255.0
            success_frames.append(frame)

        self.success_frames = np.asarray(success_frames)
        self.success_pred.backward(self.success_frames)
        self.success_pred.construct_model()

        #load the reference qpos
        self.qpos_ref = np.loadtxt(os.path.join(task_dir, 'qpos_original.txt'))


        if self.obs_mode == 'ae_feats':
            i_dim = [128, 128, 3]
            recon_dim = [32, 32, 3]
            arch = 'cnn'
            n_filters = [64, 32, 16]
            strides = [2, 1, 1]
            decoder_layers = [200, 100, 50]

            self.ae_model = AutoEncoderNetwork(arch, i_dim, recon_dim, n_filters, strides, decoder_layers)
            self.ae_model.load_wt(self.sess, AE_MODEL_PATH)
            

        model = rope(num_beads=self.num_beads, 
                    init_pos=self.init_pos,
                    texture=self.texture)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

        low = np.asarray(4*[-0.4])
        high = np.asarray(4*[0.4])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.last_reward = False


        #when using xmls        
        #mujoco_env.MujocoEnv.__init__(self, 'rope.xml', 5)

    def step(self, a):

        video_frames = self.push(a)
        img = self.sim.render(self.width, self.height, camera_name="overheadcam")/255.0
        prediction = self.success_pred.forward(img)

        movement_1 = -1.0*np.linalg.norm(self.sim.data.qpos[:2] - a[:2])
        movement_2 =  -1.0*np.linalg.norm(a[:2] - a[2:])
        action_penalty = movement_1 + movement_2

        if self.reward_mode == 'thresh':
            if self.double_frame_check:
                if prediction[0,1] > self.success_thresh:
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
                if prediction[0,1] > self.success_thresh:
                    reward = 1.0
                    is_success_cls = True
                else:
                    reward = 0.0
                    is_success_cls = False
        
        elif self.reward_mode == 'logprobs':
            prediction[prediction > 0.98] = 1.0  
            reward = np.log(prediction[0,1])
            if prediction[0,1] > self.success_thresh:
                is_success_cls = True
            else:
                is_success_cls = False
        else:
            raise NotImplementedError
        
        reward += action_penalty*self.action_penalty_const

        ob = self._get_obs()
        done = False

        #success is determined by classifier as of now
        if prediction[0,1] > 0.5:
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

        elif self.obs_mode == 'meta_feats':
            img = self.sim.render(self.width, self.height, camera_name="overheadcam")/255.0
            feats = self.success_pred.get_feats(img)[0]

            obs = np.concatenate([
                feats,
                self.sim.data.qpos.flat[:6],
                self.sim.data.qvel.flat[:6],])

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
    # env = RopeMetaClassifierEnv(texture=True, obs_mode='meta_feats')
    # bla = env.step(np.zeros(4,))
    
    env = RopeMetaClassifierEnv(texture=True, obs_mode='ae_feats')
    bla = env.step(np.zeros(4,))
    import IPython; IPython.embed()

    img = env.sim.render(env.width, env.height, camera_name="overheadcam")/255.0

    import matplotlib.pyplot as plt
    import IPython; IPython.embed()
    pass