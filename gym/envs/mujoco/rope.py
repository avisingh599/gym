import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import error, spaces

from gym.envs.mujoco.dynamic_mjc.rope import rope

import mujoco_py

class RopeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 num_beads=7, 
                 substeps=50, 
                 log_video=False, 
                 video_substeps = 5, 
                 video_h=500, 
                 video_w=500,
                 camera_name='overheadcam'):
        utils.EzPickle.__init__(self)
        
        #sim params
        self.substeps = substeps # number of intermediate positions to generate
        
        #env params
        self.num_beads = num_beads

        #video params
        self.log_video = log_video
        self.video_substeps = video_substeps
        self.video_h = video_h
        self.video_w = video_w
        self.camera_name = camera_name

        model = rope(num_beads=self.num_beads)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)
        
        low = np.asarray(4*[-0.5])
        high = np.asarray(4*[0.5])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        #import IPython; IPython.embed()
        #mujoco_env.MujocoEnv.__init__(self, 'rope.xml', 5)

    def step(self, a):
        #vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        #vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        # reward_near = - np.linalg.norm(vec_1)
        # reward_dist = - np.linalg.norm(vec_2)
        # reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        
        first_bead = self.get_body_com("bead_0")[:2]
        last_bead = self.get_body_com("bead_{}".format(self.num_beads-1))[:2]
        
        vec_1 = first_bead - last_bead
        vec_2 = np.asarray([1.0, 0.0]) #horizontal line

        cosine = np.dot(vec_1, vec_2)/(np.linalg.norm(vec_1) + 1e-10)
        abs_cos = np.abs(cosine)
        reward = abs_cos

        #import IPython; IPython.embed()
        #enquire the qpos here, and then do the interpolation thing
        #self.do_simulation(a, self.frame_skip)
        #video_frames = self.do_pos_simulation_with_substeps(a)
        
        video_frames = self.pick_place(a)
        ob = self._get_obs()
        done = False
        is_success = False #TODO maybe fix this at some point

        # if reward_dist > -0.17:
        #     is_success = True

        return ob, reward, done, dict(is_success=is_success, video_frames=video_frames)

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
                video_frames[int(i/self.video_substeps)] = env.sim.render(self.video_h, self.video_w, camera_name=self.camera_name)

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
    
    log_video = True

    #test drive the env    
    # import matplotlib.pyplot as plt
    # time_wait = 1.5
    # torque_max = 10.0
    # torque_neutral = 0.0
    # max_z = 0.05

    # actions = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.00],
    #                 [0.0, 0.0, -0.2, 0.0, 0.00],

    #                 [0.0, 0.0, -0.2, 0.0, torque_max],
    #                 [0.0, 0.0, -0.2, 0.0, torque_max],
    #                 [0.0, 0.0, -0.2, 0.0, torque_max],

    #                 [0.0, 0.0, max_z,  0.0, torque_max],
    #                 [0.0, 0.0, max_z,  0.0, torque_max],
    #                 [0.0, 0.0, max_z,  0.0, torque_max],

    #                 [0.2, 0.0, max_z,  0.0, torque_max],
    #                 [0.2, 0.0, max_z,  0.0, torque_max],
    #                 [0.2, 0.0, max_z,  0.0, torque_max],

    #                 [0.0, -0.2, max_z, 0.0, torque_neutral],
    #                 [0.0, -0.2, max_z, 0.0, torque_neutral],

    #                 ])

    # np.set_printoptions(precision=3)


    # for i in range(actions.shape[0]):
    #     act = actions[i]
    #     o,r,d,info = env.step(act)
    #     #video = np.concatenate([video, info['video_frames']])
    #     if info['video_frames'] is not None:
    #         for j in range(info['video_frames'].shape[0]):
    #             writer.writeFrame(info['video_frames'][j])

    #     plt.figure(1); plt.clf()
    #     bla = env.sim.render(500, 500, camera_name='maincam')
    #     plt.axis('off')
    #     plt.imshow(bla)
    #     plt.pause(time_wait)
    #     #env._get_viewer().render()
    #     print(env.sim.data.qpos[:actions.shape[1]])

    # if writer is not None:
    #     writer.close()

    #overheadcam, miancam, leftcam
    env = RopeEnv(num_beads=7, substeps=50, log_video=log_video, camera_name='maincam')
    video_frames = []
    # video_frames += env.pick_place(np.asarray([0.,0.,-0.2,0.2]))
    # video_frames += env.pick_place(np.asarray([-0.2,0.2,-0.2,0.2]))
    # video_frames += env.pick_place(np.asarray([0.,0.,0.2,0.2]))
    
    o,r,done,info = env.step(np.asarray([0.,0.,0.0,0.0]))
    video_frames += info['video_frames']
    print('reward: {}'.format(r))
    o,r,done,info = env.step(np.asarray([0.,0.,0.2,0.2]))
    video_frames += info['video_frames']
    print('reward: {}'.format(r))
    o,r,done,info = env.step(np.asarray([-0.2,0.2,-0.2,0.2]))
    video_frames += info['video_frames']
    print('reward: {}'.format(r))
    o,r,done,info = env.step(np.asarray([0.,0.,0.2,0.2]))
    video_frames += info['video_frames']
    print('reward: {}'.format(r))
    #import IPython; IPython.embed()

    if log_video: 
        import datetime, skvideo.io
        video_name = 'logs/rope_{}.mp4'.format(datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
        writer = skvideo.io.FFmpegWriter(video_name)

        for i in range(len(video_frames)):
            for j in range(video_frames[0].shape[0]):
                 writer.writeFrame(video_frames[i][j])
        writer.close()
