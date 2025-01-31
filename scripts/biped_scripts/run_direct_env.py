# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Isaac-Hector-Direct-v0
"""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--log", action="store_true", default=False, help="Log the environment.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.dict import print_dict



def main():
    """Main function."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # setup gym style environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env.metadata["render_fps"] = int(100/2)
    
    log_dir = os.path.join(os.path.dirname(__file__), "logs", "friction_patch")
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "mpc"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # for logging
    tag = "play"
    state_log_dir = os.path.join(log_dir, tag, "state")
    obs_log_dir = os.path.join(log_dir, tag, "obs")
    episode_length_log_dir = os.path.join(log_dir, tag, "episode_length")
    if args_cli.log:
        os.makedirs(state_log_dir, exist_ok=True)
        os.makedirs(obs_log_dir, exist_ok=True)
        os.makedirs(episode_length_log_dir, exist_ok=True)
    
    state_log = []
    obs_log = []
    episode_length_log = []
    
    trial_counter = 0
    max_trials = 100
    step_counter = 0
    max_steps = int(env_cfg.episode_length_s/(env_cfg.sim.dt * env_cfg.decimation))
    
    # reset env and mpc
    obs, _ = env.reset()

    # Simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            action = torch.zeros((args_cli.num_envs, env_cfg.action_space),dtype=torch.float32, device=args_cli.device) # type: ignore
            obs, _, terminated, time_out, _ = env.step(action)
        
        if step_counter >= max_steps-1:
            if args_cli.log:
                np.save(os.path.join(state_log_dir, f"{trial_counter}.npy"), np.array(state_log))
                np.save(os.path.join(obs_log_dir, f"{trial_counter}.npy"), np.array(obs_log))
                np.save(os.path.join(episode_length_log_dir, f"{trial_counter}.npy"), np.array(episode_length_log))
            state_log = []
            obs_log = []
            episode_length_log = []
            trial_counter += 1
            step_counter = 0
        
        if trial_counter > max_trials-1:
            print("[INFO] Max trials reached.")
            break
        
        # merge terminated and time_out
        terminated = terminated | time_out
        
        state_log.append(env.unwrapped._state.cpu().numpy()) # type: ignore
        obs_log.append(obs["policy"].cpu().numpy())
        episode_length_log.append(env.unwrapped.episode_length_buf.cpu().numpy()) # type: ignore
        
        step_counter += 1
            
    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()