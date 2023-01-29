import os
import numpy as np
from gym import utils
from wgcsl.envs import fetch_env


REACH_MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')
class FetchReachOODEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self,goal_type, ood_g_range=[0.,0.15], reward_type='sparse', target_range=0.15):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        fetch_env.FetchEnv.__init__(
            self, REACH_MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=target_range, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, goal_type=goal_type, 
            ood_g_range=ood_g_range, initial_type='', ood_obj_range=[0,0.15], env_name='Reach')
        utils.EzPickle.__init__(self)



# Ensure we get the path separator correct on windows
PUSH_MODEL_XML_PATH = os.path.join('fetch', 'push.xml')
class FetchPushOODEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, goal_type, initial_type='right', ood_g_range=[0.,0.15], ood_obj_range=[0,0.15],
                reward_type='sparse', obj_range=0.15, target_range=0.15):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, PUSH_MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=obj_range, target_range=target_range, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, goal_type=goal_type, 
            ood_g_range=ood_g_range, ood_obj_range=ood_obj_range, initial_type=initial_type, 
            env_name='Push')
        utils.EzPickle.__init__(self)


# Ensure we get the path separator correct on windows
PICK_MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')
class FetchPickOODEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, goal_type, initial_type='', ood_g_range=[0.,0.65], ood_obj_range=[0,0.15], 
                obj_range=0.15, target_range=0.15, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, PICK_MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=obj_range, target_range=target_range, distance_threshold=0.05,
            goal_type=goal_type, ood_g_range=ood_g_range, ood_obj_range=ood_obj_range, 
            initial_type=initial_type, initial_qpos=initial_qpos, reward_type=reward_type,
            env_name='Pick')
        utils.EzPickle.__init__(self)



# Ensure we get the path separator correct on windows
SLIDE_MODEL_XML_PATH = os.path.join('fetch', 'slide.xml')
class FetchSlideOODEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, goal_type, initial_type='', ood_g_range=[-0.3,0.2], ood_obj_range=[0,0.15], 
                obj_range=0.1, target_range=0.2, reward_type='sparse'): 
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.7, 1.1, 0.41, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, SLIDE_MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]), 
            obj_range=obj_range, target_range=target_range, distance_threshold=0.05,
            goal_type=goal_type, ood_g_range=ood_g_range, ood_obj_range=ood_obj_range, initial_type=initial_type, 
            initial_qpos=initial_qpos, reward_type=reward_type, env_name='Slide')
        utils.EzPickle.__init__(self)
