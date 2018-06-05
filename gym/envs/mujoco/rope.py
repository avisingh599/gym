import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py

class RopeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, 
                 substeps=50, 
                 log_video=False, 
                 video_substeps = 5, 
                 video_h=500, 
                 video_w=500,
                 camera_name='overheadcam'):
        utils.EzPickle.__init__(self)
        self.substeps = substeps # number of intermediate positions to generate
        self.log_video = log_video
        self.video_substeps = video_substeps
        self.video_h = video_h
        self.video_w = video_w
        self.camera_name = camera_name
        mujoco_env.MujocoEnv.__init__(self, 'rope.xml', 5)

    def step(self, a):
        #vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        #vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        # reward_near = - np.linalg.norm(vec_1)
        # reward_dist = - np.linalg.norm(vec_2)
        # reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        reward = 0.0
        #import IPython; IPython.embed()
        #enquire the qpos here, and then do the interpolation thing
        #self.do_simulation(a, self.frame_skip)
        video_frames = self.do_pos_simulation_with_substeps(a)
        
        ob = self._get_obs()
        done = False

        is_success = False
        # if reward_dist > -0.17:
        #     is_success = True

        return ob, reward, done, dict(is_success=is_success, video_frames=video_frames)

    def do_pos_simulation_with_substeps(self, a):
        #import IPython; IPython.embed()
        qpos_curr = self.sim.data.qpos[:self.action_space.shape[0]]
        step_size = (a - qpos_curr) / self.substeps
        
        if self.log_video:
            video_frames = np.zeros((int(self.substeps/self.video_substeps), self.video_h, self.video_w, 3))
        else:
            video_frames = None
        for i in range(self.substeps):
            self.sim.data.ctrl[:] = qpos_curr + (i+1)*step_size
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

        # self.goal_pos = np.asarray([0, 0])
        # while True:
        #     self.cylinder_pos = np.concatenate([
        #             self.np_random.uniform(low=-0.3, high=0, size=1),
        #             self.np_random.uniform(low=-0.2, high=0.2, size=1)])
        #     if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
        #         break

        # qpos[-4:-2] = self.cylinder_pos
        # qpos[-2:] = self.goal_pos

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
    
    #test drive the env    
    import matplotlib.pyplot as plt
    time_wait = 1.5
    #torque control on the gripper
    torque_max = 10.0
    torque_neutral = 0.0
    log_video = True

    actions = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.00],
                    [0.0, 0.0, -0.2, 0.0, 0.00],
                    [0.0, 0.0, -0.2, 0.0, torque_max],
                    [0.0, 0.0, -0.2, 0.0, torque_max],
                    [0.0, 0.0, -0.2, 0.0, torque_max],

                    [0.0, 0.0, 0.1,  0.0, torque_max],
                    [0.0, 0.0, 0.1,  0.0, torque_max],
                    [0.0, 0.0, 0.1,  0.0, torque_max],

                    [0.2, 0.0, 0.1,  0.0, torque_max],
                    [0.2, 0.0, 0.1,  0.0, torque_max],
                    [0.2, 0.0, 0.1,  0.0, torque_max],

                    [0.0, -0.2, 0.1, 0.0, torque_neutral],
                    [0.0, -0.2, 0.1, 0.0, torque_neutral],

                    # [0.2, 0.0, -0.25,  0.0, torque_max],
                    # [0.2, 0.0, -0.25,  0.0, torque_max],
                    # [0.2, 0.0, -0.25,  0.0, -torque_max],
                    # [0.2, 0.0, -0.25,  0.0, -torque_max],
                    # [0.0, -0.2, 0.1, 0.0, torque_neutral],
                    # [0.0, -0.2, 0.1, 0.0, torque_neutral],
                    ])

    np.set_printoptions(precision=3)

    if log_video: 
        import datetime, skvideo.io
        video_name = 'logs/video_{}.mp4'.format(datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
        writer = skvideo.io.FFmpegWriter(video_name)

    #overheadcam, miancam, leftcam
    env = RopeEnv(log_video=log_video, camera_name='overheadcam')

    for i in range(actions.shape[0]):
        act = actions[i]
        o,r,d,info = env.step(act)
        #video = np.concatenate([video, info['video_frames']])
        if info['video_frames'] is not None:
            for j in range(info['video_frames'].shape[0]):
                writer.writeFrame(info['video_frames'][j])

        plt.figure(1); plt.clf()
        bla = env.sim.render(500, 500, camera_name='maincam')
        plt.axis('off')
        plt.imshow(bla)
        plt.pause(time_wait)
        #env._get_viewer().render()
        print(env.sim.data.qpos[:actions.shape[1]])

    if writer is not None:
        writer.close()