import tensorflow as tf
from tensorflow.python.ops.init_ops import constant_initializer
import tensorflow.contrib.slim as slim
import numpy as np
import cv2

import gym

import os
import pickle
from copy import copy

from tqdm import tqdm

np.random.seed(0)


class AutoEncoderNetwork(object):
    def __init__(self, arch, i_dim, recon_dim, n_filters, strides, decoder_layers):
        #self.x_dim = x_dim
        #self.y_dim = y_dim
        self.i_dim = i_dim
        self.recon_dim = recon_dim
        # self.recon_h = 32
        # self.recon_w = 32
        # self.recon_c = 3

        if arch == 'fp':
            self._build_fp_network(n_filters, strides, decoder_layers)
        elif arch == 'cnn':
            self._build_conv_network(n_filters, strides, decoder_layers)
        else:
            raise NotImplementedError()

        self.trainable_variables = tf.trainable_variables()
        self.global_variables = tf.global_variables()

        self.conv_variables = []
        self.fc_variables = []

        for v in self.trainable_variables:
            if v.name[:4] == 'conv':
                self.conv_variables.append(v)
            elif v.name[:2] == 'fc':
                self.fc_variables.append(v)
            else:
                print('Non conv/fc variable', v.name)

        #self.data_dict = np.load('vgg16.npy', encoding='latin1').item()
        #SSLModel.__init__(self, x_dim, i_dim, y_dim, unsup_wt, n_filters)

    def _build_fp_network(self, n_filters, strides, decoder_layers):
        #self.x = tf.placeholder(tf.float32, [None] + self.x_dim)
        self.i = tf.placeholder(tf.float32, [None] + self.i_dim)
        self.i_recon = tf.placeholder(tf.float32, [None, self.recon_dim[0], self.recon_dim[1], self.recon_dim[2]])
        #self.y = tf.placeholder(tf.float32, [None] + self.y_dim)
        self.dropout = tf.placeholder(tf.float32, [])
        #self.labels_domain = tf.placeholder(tf.float32, [None, 2])
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')

        #conv1_w = self.data_dict['conv1_1'][0]
        #conv1_b = self.data_dict['conv1_1'][1]
        
        #VGG_MEAN = [103.939, 116.779, 123.68]
        i_input = self.i# - VGG_MEAN

        batch_norm_params = {'is_training': self.is_train, 'decay': 0.9, 'updates_collections': None, 'scale': True}
        
        # net = slim.conv2d(i_input, 64, [3,3], weights_initializer=constant_initializer(conv1_w), 
        #     biases_initializer=constant_initializer(conv1_b), stride=2, trainable=False, scope='conv1')
        # for i, n_filter in enumerate(n_filters):
        #     net = slim.conv2d(net, n_filter, [3,3], stride=1, scope='conv%d'%(i+2),
        #         normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params)

        #net = slim.conv2d(i_input, 64, [3,3], stride=2, trainable=False, scope='conv1')
        
        net = i_input
        for i, n_filter in enumerate(n_filters):
            net = slim.conv2d(net, n_filter, [3,3], 
                                stride=strides[i], 
                                scope='conv%d'%(i+1),
                                reuse=tf.AUTO_REUSE,
                                normalizer_fn=slim.batch_norm, 
                                normalizer_params=batch_norm_params)

        _, h, w, num_fp = net.get_shape()
        h,w,num_fp = int(h),int(w),int(num_fp)

        features = tf.transpose(net, [0,3,1,2])
        features = tf.reshape(features, [-1, num_fp, h*w])
        features_softmax = tf.nn.softmax(features)   
        features_vis = tf.reshape(features_softmax, [-1, num_fp, h, w])
        features_vis = tf.transpose(features_vis, [0,2,3,1])     

        map_y = np.tile(np.reshape(np.arange(h, dtype='float32'), (h,1)), (1,w))
        map_x = np.tile(np.reshape(np.arange(w, dtype='float32'), (1,w)), (h,1))
        map_y = (map_y-h/2.0)/h
        map_x = (map_x-w/2.0)/w
        map_y = np.reshape(map_y, (h*w,))
        map_x = np.reshape(map_x, (h*w,))
        map_y = tf.convert_to_tensor(map_y, dtype=tf.float32)
        map_x = tf.convert_to_tensor(map_x, dtype=tf.float32)
        
        feat_x = tf.reduce_sum(tf.multiply(features_softmax, map_x), reduction_indices=[2])
        feat_y = tf.reduce_sum(tf.multiply(features_softmax, map_y), reduction_indices=[2])

        self.feats = tf.concat([feat_x, feat_y], axis=1)
        #feats_lab, feats_ulab = tf.split(0, 2, self.feats)

        #dropout, but without scaling
        feats_drop = (self.dropout)*slim.dropout(self.feats, self.dropout, scope='dropout')
        
        #net = tf.concat(1, [feats_lab_drop, self.x])
        # net = tf.concat(1, [feats_lab, self.x])

        # net = slim.fully_connected(net, 400, activation_fn=tf.nn.relu, scope='fc1')
        # net = slim.fully_connected(net, 400, activation_fn=tf.nn.relu, scope='fc2')
        # self.y_pred = slim.fully_connected(net, self.y_dim[0], activation_fn=None, scope='fc_y_pred')

        # self.loss_sup = 1000.0*tf.reduce_mean(tf.square(self.y_pred - self.y))

        with tf.variable_scope('decoder'):
            net = feats_drop
            for i, num_units in enumerate(decoder_layers):
                net = slim.fully_connected(net, num_units, scope='fc_%d'%(i+1), reuse=tf.AUTO_REUSE)
            output = slim.fully_connected(net, int(np.prod(self.recon_dim)), activation_fn=None, scope='fc_last', reuse=tf.AUTO_REUSE)
            
            recon_output = tf.reshape(output, [-1, self.recon_dim[0], self.recon_dim[1], self.recon_dim[2]])

        #self.loss_recon = tf.reduce_mean(tf.square(recon_output - self.i_recon))
        self.loss = tf.reduce_mean(tf.square(recon_output - self.i_recon))

        #tf.summary.scalar("loss_recon", self.loss_recon)
        #tf.summary.scalar("loss_supervised", self.loss_sup)
        tf.summary.scalar("loss_total", self.loss)

        tf.summary.histogram("x_feats", feat_x)
        tf.summary.histogram("y_feats", feat_y)

        tf.summary.image("features_softmax_0", tf.concat([features_vis[:,:,:,0:1], features_vis[:,:,:,1:2], features_vis[:,:,:,2:3], features_vis[:,:,:,3:4]], axis=2))
        tf.summary.image("features_softmax_1", tf.concat([features_vis[:,:,:,4:5], features_vis[:,:,:,5:6], features_vis[:,:,:,6:7], features_vis[:,:,:,7:8]], axis=2))
        tf.summary.image("features_softmax_2", tf.concat([features_vis[:,:,:,8:9], features_vis[:,:,:,9:10], features_vis[:,:,:,10:11], features_vis[:,:,:,11:12]], axis=2))
        tf.summary.image("features_softmax_3", tf.concat([features_vis[:,:,:,12:13], features_vis[:,:,:,13:14], features_vis[:,:,:,14:15], features_vis[:,:,:,15:16]], axis=2))
        tf.summary.image("features_softmax_all", tf.reduce_sum(features_vis, reduction_indices=[3], keep_dims=True))
        tf.summary.image("img", self.i)
        #tf.summary.image("img_gray", tf.reshape(self.i_recon, [-1, self.recon_dim[0], self.recon_dim[1], self.recon_dim[2]]))
        tf.summary.image("img_recon", tf.reshape(recon_output, [-1, self.recon_dim[0], self.recon_dim[1], self.recon_dim[2]]))


    def _build_conv_network(self, n_filters, strides, decoder_layers):
        #self.x = tf.placeholder(tf.float32, [None] + self.x_dim)
        self.i = tf.placeholder(tf.float32, [None] + self.i_dim)
        self.i_recon = tf.placeholder(tf.float32, [None, self.recon_dim[0], self.recon_dim[1], self.recon_dim[2]])
        #self.y = tf.placeholder(tf.float32, [None] + self.y_dim)
        self.dropout = tf.placeholder(tf.float32, [])
        #self.labels_domain = tf.placeholder(tf.float32, [None, 2])
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')

        #conv1_w = self.data_dict['conv1_1'][0]
        #conv1_b = self.data_dict['conv1_1'][1]
        
        #VGG_MEAN = [103.939, 116.779, 123.68]
        i_input = self.i# - VGG_MEAN

        batch_norm_params = {'is_training': self.is_train, 'decay': 0.9, 'updates_collections': None, 'scale': True}

        net = i_input
        for i, n_filter in enumerate(n_filters):
            net = slim.conv2d(net, n_filter, [3,3], 
                                stride=strides[i], 
                                scope='conv%d'%(i+1),
                                reuse=tf.AUTO_REUSE,
                                normalizer_fn=slim.batch_norm, 
                                normalizer_params=batch_norm_params)

        with tf.variable_scope('decoder'):
            net = slim.flatten(net)
            for i, num_units in enumerate(decoder_layers):
                net = slim.fully_connected(net, num_units, scope='fc_%d'%(i+1), reuse=tf.AUTO_REUSE)
            self.feats = net
            output = slim.fully_connected(net, int(np.prod(self.recon_dim)), activation_fn=None, scope='fc_last', reuse=tf.AUTO_REUSE)
            
            recon_output = tf.reshape(output, [-1, self.recon_dim[0], self.recon_dim[1], self.recon_dim[2]])

        #self.loss_recon = tf.reduce_mean(tf.square(recon_output - self.i_recon))
        self.loss = tf.reduce_mean(tf.square(recon_output - self.i_recon))

        #tf.summary.scalar("loss_recon", self.loss_recon)
        #tf.summary.scalar("loss_supervised", self.loss_sup)
        tf.summary.scalar("loss_total", self.loss)
        tf.summary.image("img", self.i, max_outputs=8)
        #tf.summary.image("img_gray", tf.reshape(self.i_recon, [-1, self.recon_dim[0], self.recon_dim[1], self.recon_dim[2]]))
        tf.summary.image("img_recon", 
            tf.reshape(recon_output, 
                [-1, self.recon_dim[0], self.recon_dim[1], self.recon_dim[2]]),
            max_outputs=8)

    def train(self, sess, data_train, data_val, exp, seed=0, steps=int(10e4), 
        save_frq=5000, print_frq=100, learning_rate=3e-4, batch_size=32, 
        log_dir='log/', save_dir='checkpoints/', evaluate=False):

        tf.set_random_seed(seed)
        np.random.seed(seed)

        opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        #opt_init = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_recon)

        sess.run(tf.global_variables_initializer())

        import datetime
        time_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_dir_tr = log_dir + time_string + '_' + exp + '_train/'
        log_dir_va = log_dir + time_string + '_' + exp + '_valid/'          
        save_dir = save_dir + time_string + '_' + exp + '/'

        merged_summary_op = tf.summary.merge_all()
        
        summary_writer_train = tf.summary.FileWriter(log_dir_tr, graph_def=sess.graph_def)
        summary_writer_valid = tf.summary.FileWriter(log_dir_va, graph_def=sess.graph_def)

        for i in tqdm(range(steps)):
            #if i>steps_init:
            loss, _ = sess.run([self.loss, opt], feed_dict=self._get_batch(batch_size, data_train))
            #else:
            #    loss, _ = sess.run([self.loss_sup, opt_init], feed_dict=self._get_batch(batch_size, data_train))

            if (i+1)%print_frq == 0:
                summary_str = sess.run(merged_summary_op, feed_dict=self._get_all_data(data_val))
                summary_writer_valid.add_summary(summary_str, i+1)
                summary_str = sess.run(merged_summary_op, feed_dict=self._get_all_data(data_train))
                summary_writer_train.add_summary(summary_str, i+1)

            if (i+1)%save_frq == 0:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file = save_dir+('weights_%06d.pkl'%(i+1))
                self._save_chkpt(sess, save_file)
                #if evaluate:
                #    os.system("python eval_img.py %s"%(save_file))

    def _get_batch(self, batch_size, data, is_train=True):
        #num_points = data['X'].shape[0]
        #batch_ind = np.random.randint(num_points, size=batch_size)

        #num_points_ext = data['I_ext_1'].shape[0]
        #batch_ind_ext_1 = np.random.randint(num_points_ext, size=batch_size)
        #img_batch = np.concatenate([data['I'][batch_ind], 
                        #data['I_ext_1'][batch_ind_ext_1]], axis=0)
        
        num_points = data.shape[0]
        batch_ind = np.random.randint(num_points, size=batch_size)
        img_batch = data[batch_ind]

        i_recon = np.zeros((batch_size, self.recon_dim[0], self.recon_dim[1], self.recon_dim[2]))

        for i in range(batch_size):
            i_recon[i] = cv2.resize(img_batch[i], (self.recon_dim[0],self.recon_dim[1]))

        dropout = 0.5

        feed_dict = {
                self.is_train: is_train, 
                self.dropout: dropout,
                #self.x : data['X'][batch_ind], 
                self.i : img_batch, 
                #self.y : data['y'][batch_ind], 
                self.i_recon : i_recon,
                }
        
        return feed_dict

    def _get_all_data(self, data):
        upper_limit = 250
        if data.shape[0] > upper_limit:
            return self._get_batch(upper_limit, data, is_train=False)
        else:
            raise NotImplementedError

    def _save_chkpt(self, sess, savefile):
        variables = {}
        for v in self.trainable_variables:
            variables[v.name] = sess.run(v)

        for v in self.global_variables:
            if 'BatchNorm' in v.name:
                variables[v.name] = sess.run(v)

        pickle.dump(variables, open(savefile, 'wb'))

        print('Saved weights to {}'.format(savefile))

    def load_wt(self, sess, wt_file):
        sess.run(tf.global_variables_initializer())
        val_vars = pickle.load(open(wt_file, 'rb'))

        for v in self.trainable_variables:
            if v.name in val_vars:
                assign_op = v.assign(val_vars[v.name])
                sess.run(assign_op)
            else:
                print('Variable not loaded: ', v.name)

        for v in self.global_variables:
            if 'BatchNorm' in v.name:
                assign_op = v.assign(val_vars[v.name])
                sess.run(assign_op)


def collect_data(env, num_traj, num_timesteps, i_dim):
    
    #import IPython; IPython.embed()

    images_all = np.zeros((num_traj*num_timesteps, i_dim[0], i_dim[1], i_dim[2])) 
    counter = 0
    i_dims = i_dim[0]*i_dim[1]*i_dim[2]
    for _ in range(num_traj):
        ob = env.reset()
        for _ in range(num_timesteps):
            rand_act = np.random.uniform(low=env.action_space.low, high=env.action_space.high)
            ob, r, done, _ = env.step(rand_act)
            images_all[counter] = np.reshape(ob[:i_dims], (i_dim[0], i_dim[1], i_dim[2]))
            counter+=1

    return images_all

if __name__ == "__main__": 

    #fp architecture
    # arch = 'fp'
    # n_filters = [64, 32, 32]
    # strides = [2, 1, 1]
    # i_dim = [64, 64, 3]
    # recon_dim = [32, 32, 3]
    # decoder_layers = [100, 100]

    #cnn architecture
    arch = 'cnn'
    n_filters = [64, 32, 16]
    strides = [2, 1, 1]
    i_dim = [64, 64, 3]
    recon_dim = [32, 32, 3]
    decoder_layers = [200, 100, 50]

    num_traj_tr = 100
    num_traj_va = 20
    num_timesteps = 100
    
    #env = PusherVisionEnv(randomized=True, frame_skip=5)
    #env = TfEnv(CustomGymEnv('Pusher7DOFVision-v0'))
    env = gym.make('PusherVision-v0')

    data_tr = collect_data(env, num_traj_tr, num_timesteps, i_dim)
    data_va = collect_data(env, num_traj_va, num_timesteps, i_dim)

    model = AutoEncoderNetwork(arch, i_dim, recon_dim, n_filters, strides, decoder_layers)

    filter_string = ''
    for f in n_filters: 
        filter_string += str(f) + '_'

    exp = 'cv2_mjc150_dropout_bn_{}_filters_{}'.format(arch, filter_string)
    
    with tf.Session() as sess:
        model.train(sess, data_tr, data_va, exp, 
                    save_dir='ae_checkpoints/', log_dir='ae_log/',
                    steps=int(5e4))

    #seems to be working so far. need to train the model next
