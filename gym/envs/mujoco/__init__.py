from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco.humanoid import HumanoidEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym.envs.mujoco.reacher import ReacherEnv
from gym.envs.mujoco.swimmer import SwimmerEnv
from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from gym.envs.mujoco.pusher import PusherEnv
from gym.envs.mujoco.thrower import ThrowerEnv
from gym.envs.mujoco.striker import StrikerEnv
from gym.envs.mujoco.reacher_vision import ReacherVisionEnv
from gym.envs.mujoco.pusher_vision import PusherVisionEnv
from gym.envs.mujoco.pusher_blind import PusherBlindEnv

#1D vision, small res stuff
from gym.envs.mujoco.pusher_1Dvision import Pusher1DVisionEnv
from gym.envs.mujoco.pusher_vision_tiny import PusherVisionTinyEnv

#reacher stuff
from gym.envs.mujoco.reacher_no_velocity_fixed_goal import ReacherNoVelocityFixedGoalEnv
from gym.envs.mujoco.reacher_vision_only import ReacherVisionOnlyEnv
from gym.envs.mujoco.reacher_discrete_no_velocity_fixed_goal import ReacherDiscreteNoVelocityFixedGoalEnv

#discretization
from gym.envs.mujoco.pusher_discrete import PusherDiscreteEnv

#autoenc envs
from gym.envs.mujoco.pusher_vision_autoencoder import PusherVisionAutoEncoderEnv

from gym.envs.mujoco.simple_reacher import SimpleReacherEnv
from gym.envs.mujoco.simple_reacher_v2 import SimpleReacherEnvV2
from gym.envs.mujoco.multi_reacher import MultiReacherEnv
from gym.envs.mujoco.multi_reacher_v2 import MultiReacherEnvV2
#from gym.envs.mujoco.simple_pusher import SimplePusherEnv

#classifier envs
from gym.envs.mujoco.multi_reacher_meta_classifier import MultiReacherMetaClassifierEnv


#rope stuff
from gym.envs.mujoco.rope import RopeEnv
from gym.envs.mujoco.rope_meta_classifier import RopeMetaClassifierEnv
from gym.envs.mujoco.rope_oracle import RopeOracleEnv
from gym.envs.mujoco.rope_ae import RopeAEEnv
