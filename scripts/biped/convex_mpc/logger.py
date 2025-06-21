import os
from typing import Literal, List, Dict
import numpy as np
import pickle
import torch

class BenchmarkLogger:
    def __init__(self, log_dir:str, tag:str, num_envs:int=1, max_trials:int=1, max_episode_length:int=1):
        self.log_dir = log_dir
        self.tag = tag
        self.num_envs = num_envs
        self.max_trials = max_trials
        self.max_episode_length = max_episode_length
        
        self.state_log_dir = os.path.join(log_dir, "logs", tag, "state")
        self.obs_log_dir = os.path.join(log_dir, "logs", tag, "obs")
        self.raw_action_log_dir = os.path.join(log_dir, "logs", tag, "raw_action")
        self.action_log_dir = os.path.join(log_dir, "logs", tag, "action")
        self.rft_log_dir = os.path.join(log_dir, "logs", tag, "rft")
        self.done_log_dir = os.path.join(log_dir, "logs", tag, "episode")
        
        os.makedirs(self.state_log_dir, exist_ok=True)
        os.makedirs(self.obs_log_dir, exist_ok=True)
        os.makedirs(self.raw_action_log_dir, exist_ok=True)
        os.makedirs(self.action_log_dir, exist_ok=True)
        os.makedirs(self.rft_log_dir, exist_ok=True)
        os.makedirs(self.done_log_dir, exist_ok=True)
        
        # temporary logs used in the program
        self.state_log = [np.array([])]*num_envs
        self.rft_log = [np.array([])]*num_envs
        self.obs_log = [np.array([])]*num_envs
        self.raw_action_log = [np.array([])]*num_envs
        self.action_log = [np.array([])]*num_envs
        self.done_log = [np.array([])]*num_envs

        # used to save as pkl
        # 2d list (max_trials, num_envs)
        self.state_log_buffer = [[np.array([]) for _ in range(self.num_envs)] for _ in range(max_trials)]
        self.rft_log_buffer = [[np.array([]) for _ in range(self.num_envs)] for _ in range(max_trials)]
        self.obs_log_buffer = [[np.array([]) for _ in range(self.num_envs)] for _ in range(max_trials)]
        self.raw_action_log_buffer = [[np.array([]) for _ in range(self.num_envs)] for _ in range(max_trials)]
        self.action_log_buffer = [[np.array([]) for _ in range(self.num_envs)] for _ in range(max_trials)]
        self.done_log_buffer = [[np.array([]) for _ in range(self.num_envs)] for _ in range(max_trials)]
        self.episode_length_buffer = [[0 for _ in range(self.num_envs)] for _ in range(max_trials)]
    

    def log(self, state:np.ndarray|None=None, 
            obs:np.ndarray|None=None, 
            raw_action:np.ndarray|None=None, 
            action:np.ndarray|None=None, 
            rft:np.ndarray|None=None, 
            done:np.ndarray|None=None):
        """
        state (num_envs, state_dim)
        obs (num_envs, obs_dim)
        raw_action (num_envs, action_dim)
        action (num_envs, action_dim)
        rft (num_envs, num_feet, num_rft_features)
        done (num_envs)
        """
        for idx in range(self.num_envs):
            if state is not None:
                if len(self.state_log[idx])==0:
                    self.state_log[idx] = state[idx]
                else:
                    self.state_log[idx] = np.vstack([self.state_log[idx], state[idx]])
            if obs is not None:
                if len(self.obs_log[idx])==0:
                    self.obs_log[idx] = obs[idx]
                else:
                    self.obs_log[idx] = np.vstack([self.obs_log[idx], obs[idx]])
            if raw_action is not None:
                if len(self.raw_action_log[idx])==0:
                    self.raw_action_log[idx] = raw_action[idx]
                else:
                    self.raw_action_log[idx] = np.vstack([self.raw_action_log[idx], raw_action[idx]])
            if action is not None:
                if len(self.action_log[idx])==0:
                    self.action_log[idx] = action[idx]
                else:
                    self.action_log[idx] = np.vstack([self.action_log[idx], action[idx]])
            if rft is not None:
                if len(self.rft_log[idx])==0:
                    self.rft_log[idx] = rft[idx][None, :, :]
                else:
                    self.rft_log[idx] = np.concatenate([self.rft_log[idx], rft[idx][None, :, :]], axis=0)
    
    def save_to_buffer(self, trial_id:int=0, env_idx:int=0):
        self.state_log_buffer[trial_id][env_idx] = self.fill_missing_data(self.state_log[env_idx])
        self.obs_log_buffer[trial_id][env_idx] = self.fill_missing_data(self.obs_log[env_idx])
        self.raw_action_log_buffer[trial_id][env_idx] = self.fill_missing_data(self.raw_action_log[env_idx])
        self.action_log_buffer[trial_id][env_idx] = self.fill_missing_data(self.action_log[env_idx])
        self.rft_log_buffer[trial_id][env_idx] = self.fill_missing_data(self.rft_log[env_idx])
        self.reset_buffer(env_idx)
    
    def fill_missing_data(self, data):
        num_dim = len(data.shape)
        if num_dim==1:
            return np.concatenate([data, np.zeros(self.max_episode_length - len(data))], axis=0)
        elif num_dim==2:
            return np.concatenate([data, np.tile(np.zeros_like(data[0:1]), (self.max_episode_length - len(data), 1))], axis=0)
        elif num_dim==3:
            return np.concatenate([data, np.tile(np.zeros_like(data[0:1]), (self.max_episode_length - len(data), 1, 1))], axis=0)
        else:
            raise ValueError("Invalid data shape")
    
    def save_episode_length_to_buffer(self, trial_id:int=0, env_idx:int=0, episode_length:int=0):
        self.episode_length_buffer[trial_id][env_idx] = episode_length
    
    def reset_buffer(self, env_idx:int=0):
        self.state_log[env_idx] = np.array([])
        self.obs_log[env_idx] = np.array([])
        self.raw_action_log[env_idx] = np.array([])
        self.action_log[env_idx] = np.array([])
        self.rft_log[env_idx] = np.array([])
        self.done_log[env_idx] = np.array([])
    
    def save(self):
        "Save the entire result"
        with open(os.path.join(self.state_log_dir, "state.pkl"), "wb") as f:
            pickle.dump(self.state_log_buffer, f)
        with open(os.path.join(self.obs_log_dir, "obs.pkl"), "wb") as f:
            pickle.dump(self.obs_log_buffer, f)
        with open(os.path.join(self.raw_action_log_dir, "raw_action.pkl"), "wb") as f:
            pickle.dump(self.raw_action_log_buffer, f)
        with open(os.path.join(self.action_log_dir, "action.pkl"), "wb") as f:
            pickle.dump(self.action_log_buffer, f)
        with open(os.path.join(self.rft_log_dir, "rft.pkl"), "wb") as f:
            pickle.dump(self.rft_log_buffer, f)
        with open(os.path.join(self.done_log_dir, "episode_length.pkl"), "wb") as f:
            pickle.dump(self.episode_length_buffer, f)
            


class DictBenchmarkLogger:
    def __init__(self, 
                 log_dir:str, 
                 tag:str, 
                 num_envs:int, 
                 max_trials:int, 
                 max_episode_length:int, 
                 log_item: List[str]):
        self.log_dir = log_dir
        self.tag = tag
        self.num_envs = num_envs
        self.max_trials = max_trials
        self.max_episode_length = max_episode_length
        self.log_item = log_item
        
        self.item_log_dir = {}
        self.item_log_cache = {}
        self.item_log_buffer = {}
        
        for item in log_item:
            self.item_log_dir[item] = os.path.join(log_dir, "logs", tag, item)
            os.makedirs(self.item_log_dir[item], exist_ok=True)
            
            # temporary log cache
            self.item_log_cache[item] = [np.array([])]*num_envs
            
            # buffer that is saved as pkl
            # 2d list: (max_trials, num_envs)
            self.item_log_buffer[item] = [[np.array([]) for _ in range(self.num_envs)] for _ in range(max_trials)]
        
        # episode length 
        self.episode_length_log_dir = os.path.join(log_dir, "logs", tag, "episode")
        os.makedirs(self.episode_length_log_dir, exist_ok=True)
        self.episode_length_buffer = [[0 for _ in range(self.num_envs)] for _ in range(max_trials)]
        
        
    def log(self, item_dict: Dict[str, np.ndarray|None]):
        """
        item_dict: Dictionary containing the items to log.
        Each key should be one of the log items specified during initialization.
        The value should be a numpy array or None.
        """
        for idx in range(self.num_envs):
            for item, data in item_dict.items():
                if data is not None:
                    if len(data.shape) == 1:
                        data = data[:, None] # add a new axis if data is 1D
                    if len(self.item_log_cache[item][idx]) == 0:
                        self.item_log_cache[item][idx] = data[idx]
                    else:
                        self.item_log_cache[item][idx] = np.vstack([self.item_log_cache[item][idx], data[idx]]) # append new data
                        
    def save_to_buffer(self, trial_id:int=0, env_idx:int=0):
        for item in self.log_item:
            self.item_log_buffer[item][trial_id][env_idx] = self.fill_missing_data(self.item_log_cache[item][env_idx])
        self.reset_buffer(env_idx)
    
    def reset_buffer(self, env_idx:int=0):
        for item in self.log_item:
            self.item_log_cache[item][env_idx] = np.array([])
    
    def fill_missing_data(self, data):
        num_dim = len(data.shape)
        if num_dim==1:
            return np.concatenate([data, np.zeros(self.max_episode_length - len(data))], axis=0)
        elif num_dim==2:
            return np.concatenate([data, np.tile(np.zeros_like(data[0:1]), (self.max_episode_length - len(data), 1))], axis=0)
        elif num_dim==3:
            return np.concatenate([data, np.tile(np.zeros_like(data[0:1]), (self.max_episode_length - len(data), 1, 1))], axis=0)
        else:
            raise ValueError("Invalid data shape")
    
    def save_episode_length_to_buffer(self, trial_id:int=0, env_idx:int=0, episode_length:int=0):
        self.episode_length_buffer[trial_id][env_idx] = episode_length
        
    def save(self):
        for item in self.log_item:
            with open(os.path.join(self.item_log_dir[item], f"{item}.pkl"), "wb") as f:
                pickle.dump(self.item_log_buffer[item], f)
        
        # episode length
        with open(os.path.join(self.episode_length_log_dir, "episode_length.pkl"), "wb") as f:
            pickle.dump(self.episode_length_buffer, f)