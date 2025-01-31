import os
from typing import Literal
import numpy as np
import pickle
import torch

class BenchmarkLogger:
    def __init__(self, log_dir:str, tag:str):
        self.log_dir = log_dir
        self.tag = tag
        
        self.state_log_dir = os.path.join(log_dir, "logs", tag, "state")
        self.obs_log_dir = os.path.join(log_dir, "logs", tag, "obs")
        self.raw_action_log_dir = os.path.join(log_dir, "logs", tag, "raw_action")
        self.action_log_dir = os.path.join(log_dir, "logs", tag, "action")
        self.rft_log_dir = os.path.join(log_dir, "logs", tag, "rft")
        self.done_log_dir = os.path.join(log_dir, "logs", tag, "done")
        
        os.makedirs(self.state_log_dir, exist_ok=True)
        os.makedirs(self.obs_log_dir, exist_ok=True)
        os.makedirs(self.raw_action_log_dir, exist_ok=True)
        os.makedirs(self.action_log_dir, exist_ok=True)
        os.makedirs(self.rft_log_dir, exist_ok=True)
        os.makedirs(self.done_log_dir, exist_ok=True)
        
        self.state_log = []
        self.rft_log = []
        self.obs_log = []
        self.raw_action_log = []
        self.action_log = []
        self.done_log = []
    
    
    def log(self, state:np.ndarray=None, obs:np.ndarray=None, raw_action:np.ndarray=None, action:np.ndarray=None, rft:np.ndarray=None, done:np.ndarray=None):
        if state is not None:
            self.state_log.append(state)
        if obs is not None:
            self.obs_log.append(obs)
        if raw_action is not None:
            self.raw_action_log.append(raw_action)
        if action is not None:
            self.action_log.append(action)
        if rft is not None:
            self.rft_log.append(rft)
        if done is not None:
            self.done_log.append(done)
    
    def save(self, id:int=0):
        if len(self.state_log)>0:
            with open(os.path.join(self.state_log_dir, f"state_{id}.pkl"), "wb") as f:
                pickle.dump(self.state_log, f)
        if len(self.obs_log)>0:
            with open(os.path.join(self.obs_log_dir, f"obs_{id}.pkl"), "wb") as f:
                pickle.dump(self.obs_log, f)
        if len(self.raw_action_log)>0:
            with open(os.path.join(self.raw_action_log_dir, f"raw_action_{id}.pkl"), "wb") as f:
                pickle.dump(self.raw_action_log, f)
        if len(self.action_log)>0:
            with open(os.path.join(self.action_log_dir, f"action_{id}.pkl"), "wb") as f:
                pickle.dump(self.action_log, f)
        if len(self.rft_log)>0:
            with open(os.path.join(self.rft_log_dir, f"rft_{id}.pkl"), "wb") as f:
                pickle.dump(self.rft_log, f)
        if len(self.done_log)>0:
            with open(os.path.join(self.done_log_dir, f"done_{id}.pkl"), "wb") as f:
                pickle.dump(self.done_log, f)
        self.reset_buffer()
    
    def reset_buffer(self):
        self.state_log = []
        self.rft_log = []
        self.obs_log = []
        self.raw_action_log = []
        self.action_log = []
        self.done_log = []