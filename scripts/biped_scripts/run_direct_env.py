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
parser = argparse.ArgumentParser(description="Running environment without RL")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--log", action="store_true", default=False, help="Log the environment.")
parser.add_argument("--log_extra", action="store_true", default=False, help="Log extra information.")
parser.add_argument("--tag", type=str, default=None, help="Tag for logging.")
parser.add_argument("--max_trials", type=int, default=1, help="Number of trials to run.")
parser.add_argument("--episode_length", type=float, default=20.0, help="Length of the episode in second.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")


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
import datetime
import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.dict import print_dict
from scripts.biped.rsl_rl.logger import BenchmarkLogger


def main():
    """Main function."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # setup gym style environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env.metadata["render_fps"] = int(100/2)
    
    log_dir = os.path.join(os.path.dirname(__file__), "logs", args_cli.task, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    # for logging
    if args_cli.tag:
        name = f"mpc/{args_cli.tag}"
    else:
        name = "mpc"
    
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", name),
            # "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    
    if args_cli.log:
        max_episode_length = int(env_cfg.episode_length_s/(env_cfg.decimation*env_cfg.sim.dt))
        logger = BenchmarkLogger(log_dir, name, num_envs=args_cli.num_envs, max_trials=args_cli.max_trials, max_episode_length=max_episode_length)
    max_trials = args_cli.max_trials
    episode_length_log = [0]*args_cli.num_envs
    episode_counter = [0]*args_cli.num_envs

    # reset environment
    obs, _ = env.reset()
    
    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            action = torch.zeros((args_cli.num_envs, env_cfg.num_actions),dtype=torch.float32, device=args_cli.device) # type: ignore
            obs, _, terminated, time_out, _ = env.step(action)
            dones = terminated | time_out
            obs = obs["policy"]
        
        if args_cli.log:
            if args_cli.log_extra:
                rft=torch.cat([env.unwrapped.foot_depth.unsqueeze(2),  # type: ignore
                                    env.unwrapped.foot_angle_beta.unsqueeze(2),  # type: ignore
                                    env.unwrapped.foot_angle_gamma.unsqueeze(2),  # type: ignore
                                    env.unwrapped.foot_velocity_b, # type: ignore
                                    env.unwrapped.foot_accel, # type: ignore
                                    env.unwrapped.rft_force,  # type: ignore
                                    env.unwrapped.damping_force,  # type: ignore
                                    env.unwrapped.tangential_force], dim=2) # type: ignore
                logger.log(state=env.unwrapped._state.cpu().numpy(), # type: ignore
                        obs=obs.cpu().numpy(),
                        raw_action=action.cpu().numpy(),
                        action=env.unwrapped._actions_op.cpu().numpy(), # type: ignore
                        rft=rft.cpu().numpy(), 
                        done=dones.cpu().numpy(),  # type: ignore
                        )
            else:
                logger.log(state=env.unwrapped._state.cpu().numpy(), # type: ignore
                        obs=obs.cpu().numpy(),
                        raw_action=actions.cpu().numpy(), # type: ignore  # noqa: F821
                        action=env.unwrapped._actions_op.cpu().numpy(), # type: ignore
                        done=dones.cpu().numpy(),  # type: ignore
                        )
        # Incremenet episode length 
        for i  in range(args_cli.num_envs):
            episode_length_log[i] += 1
        
        # Convert done flag to numpy
        dones_np = dones.cpu().numpy() # type: ignore

        # Check each agent's done flag
        for i, done_flag in enumerate(dones_np):
            if episode_counter[i] < max_trials: # only allow logging when episode counter is less than max trials to avoid memory overflow
                if done_flag == 1:
                    print(f"[INFO] Env {i}: Episode {episode_counter[i]} completed with episode length {episode_length_log[i]}.")
                    if args_cli.log:
                        logger.save_to_buffer(trial_id=episode_counter[i], env_idx=i)
                        logger.save_episode_length_to_buffer(trial_id=episode_counter[i], env_idx=i, episode_length=episode_length_log[i])
                    episode_length_log[i] = 0
                    episode_counter[i] += 1
        
        # Check if max trials reached
        if all(tc >=max_trials for tc in episode_counter):
            print("[INFO] Max trials reached for all environments.")
            break

    # save all the log
    if args_cli.log:
        logger.save()
        print(f"Saved logs in {logger.log_dir}")
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()