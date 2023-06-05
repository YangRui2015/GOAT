import gym
import numpy as np
from gym.core import Wrapper
from gym.spaces import Dict, Box
import copy
from numpy.linalg.linalg import norm

class FetchGoalWrapper(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env)
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
    
    def reset(self):
        return self.env.reset()
    
    def compute_rewards(self, achieved_goal, desired_goal, info=None):
        return self.env.compute_rewards(achieved_goal, desired_goal, info)
    
    def compute_reward(self, achieved_goal, desired_goal, info=None):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def step(self, action):
        return self.env.step(action)
    
    def render(self, mode='human'):
        return self.env.render()
    
    def sample_goal(self):
        import pdb;pdb.set_trace
        return self.env.env._sample_goal()



# for point env 
class PointGoalWrapper(Wrapper):
    observation_keys = ['observation', 'desired_goal', 'achieved_goal']
    def __init__(self, env):
        Wrapper.__init__(self, env=env)
        self.env = env
        self.action_space = env.action_space
        # observation
        for key in list(env.observation_space.spaces.keys()):
            if key not in self.observation_keys:
                del env.observation_space.spaces[key]

        self.observation_space = env.observation_space
        # self.env.env.reward_type = 'sparse'
        temp_env = self.env.env 
        if hasattr(temp_env, 'reward_type'):
            temp_env.reward_type = 'sparse'
        while hasattr(temp_env, 'env'):
            temp_env = temp_env.env 
            if hasattr(temp_env, 'reward_type'):
                temp_env.reward_type = 'sparse'

    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        obs = {
            'observation':obs_dict['observation'],
            'desired_goal':obs_dict['desired_goal'],
            'achieved_goal':obs_dict['achieved_goal']
        }
        return obs, reward, done, info
    
    def render(self, mode='human'):
        return self.env.render()
    
    def compute_reward(self, achieved_goal, desired_goal, info=None):
        obs = {
            'state_achieved_goal': achieved_goal,
            'state_desired_goal':desired_goal
        }
        action = np.array([])
        return self.env.compute_rewards(action, obs)

    def sample_goal(self):
        goal_dict = self.env.sample_goal()
        return goal_dict['desired_goal']
    
    def set_goal(self, goal_id):
        self.env.set_goal(goal_id)
