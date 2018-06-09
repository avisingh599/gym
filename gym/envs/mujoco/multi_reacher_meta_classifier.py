import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import gc
import glob
import os
from random import shuffle
from natsort import natsorted

import tensorflow as tf
from tensorflow.python.platform import flags
from visual_mpc.one_shot_predictor import OneShotPredictor
import imageio
import time

import argparse
tf.flags._global_parser = argparse.ArgumentParser() #hacky stuff to clear the flags
FLAGS = flags.FLAGS

#temp flags for the OneShotModel to work, remove them later
flags.DEFINE_integer('im_height', 64, 'h')
flags.DEFINE_integer('im_width', 64, 'w')

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
flags.DEFINE_bool('aux_inp', False, 'use auxiliary input')
flags.DEFINE_bool('grad_clip', True, 'use gradient clipping')
flags.DEFINE_float('clip_min', -50.0, 'minimum for gradient clipping')
flags.DEFINE_float('clip_max', 50.0, 'maximum for gradient clipping')
flags.DEFINE_bool('stop_grad', True, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Model options
flags.DEFINE_string('norm', 'layer_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_conv_layers', 3, 'number of convolutional layers')
flags.DEFINE_integer('num_filters', 16, 'number of filters for conv nets.')
flags.DEFINE_integer('num_fc_layers', 1, 'number of fully connected layers')
flags.DEFINE_integer('hidden_dim', 50, 'hidden dimension of fully connected layers')
flags.DEFINE_bool('fc_bt', False, 'use bias transformation for the first fc layer')
flags.DEFINE_integer('bt_dim', 0, 'the dimension of bias transformation for FC layers')
flags.DEFINE_bool('fp', True, 'use feature spatial soft-argmax')

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
flags.DEFINE_integer('num_tasks', 3, 'number of tasks to perform in succession')

flags.DEFINE_bool('resnet_feats', False, 'resnet_feats')
flags.DEFINE_bool('vgg_path', False, 'resnet_feats')



class MultiReacherMetaClassifierEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, object_id=6, env_id=7):
        gc.enable()
        utils.EzPickle.__init__(self)

        assert(env_id > 4)
        #import IPython; IPython.embed()
        # self.object_xml_paths = natsorted(glob.glob(os.path.join(os.path.dirname(__file__), "assets/testing3/*")))
        # self.object_xml_iter = iter(self.object_xml_paths)
        
        # self.xml_paths = natsorted(glob.glob(self.object_xml_iter.__next__() + "/*"))
        # self.xml_iter = iter(self.xml_paths)

        # mujoco_env.MujocoEnv.__init__(self, self.xml_iter.__next__(), 5)
        
        #TODO, fix this soon
        #self.pretrained_model_path
        mujoco_env.MujocoEnv.__init__(self, 'testing3/object_{}/train_{}.xml'.format(object_id, env_id), 5)

        #TODO fix this soon, make this an env variable maybe
        pretrained_model_path = '/media/avi/data/Work/proj_3/data_from_annie/pretrained_models/5_shot_50_pos/model59999'
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.success_pred = OneShotPredictor(pretrained_model_path, self.sess)

        #TODO this path also needs to be fixed
        #object_dirs = ['/media/avi/data/Work/proj_3/data_from_annie/testing/object_%d/' % i for i in range(30)]
        object_dir = '/media/avi/data/Work/proj_3/data_from_annie/testing/object_{}/'.format(object_id)
        successes = ['success_0.jpg', 'success_1.jpg', 'success_2.jpg', 'success_3.jpg', 'success_4.jpg']
        success_frames = []

        for k in range(5):
            frame = imageio.imread(object_dir + successes[k])[:, :, :3] / 255.0
            success_frames.append(frame)

        self.success_frames = np.asarray(success_frames)
        self.success_pred.backward(self.success_frames)
        self.success_pred.construct_model()

        self.last_reward = False

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
        
        #reward = reward_dist + reward_ctrl
        
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        done = False
        if reward_dist > -0.08:
            is_success=True
        else:
            is_success = False

        #classifier based reward is computed here
        width=64
        height=64

        #start = time.time()
        img = self.sim.render(width, height, camera_name="camera")/255.0
        #end = time.time()
        #print('Render time: {}'.format(end - start))

        #start = time.time()
        prediction = self.success_pred.forward(img)
        #end = time.time()
        #print('Forward time: {}'.format(end - start))
        # start = time.time()
        #with tf.variable_scope('model', reuse=None) as training_scope:
        
        #prediction = self.success_pred.predict(self.success_frames, img)
        #start = time.time()
        #import IPython; IPython.embed()
        if prediction[0,1] > 0.85:
            if self.last_reward:
                reward = 1.0
            else:
                reward = 0.0
                self.last_reward = True
        else:
            reward = 0.0
            self.last_reward = False

        #count false positive/negatives
        if reward == 1.0 and not is_success:
            false_positive=True
        else:
            false_positive=False

        if reward == 0.0 and is_success:
            false_negative=True
        else:
            false_negative=False

        #reward = np.argmax(prediction)
        #end = time.time()
        #print(end - start)
        # end = time.time()
        # print(end - start)

        return ob, reward, done, dict(reward_dist=reward_dist,
                                      reward_dist_1=reward_dist_1,
                                      reward_dist_2=reward_dist_2,
                                      reward_dist_tip=reward_dist_tip,
                                      reward_ctrl=reward_ctrl,
                                      is_success=is_success,
                                      false_positive=false_positive,
                                      false_negative=false_negative,
                                      )

    def reset_model(self):
        #print("resetted")
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

if __name__ == "__main__":
    env = MultiReacherMetaClassifierEnv()
    a = np.zeros(env.action_space.low.shape)
    #import time
    start = time.time()
    ob, r, d, inf = env.step(a)
    #start = time.time()
    end = time.time()
    print(end - start)
    #print(r)
    #import IPython; IPython.embed()
    #import IPython; IPython.embed()
