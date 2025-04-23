# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--log", action="store_true", default=False, help="Log the environment.")
parser.add_argument("--log_extra", action="store_true", default=False, help="Log extra information.")
parser.add_argument("--use_rl", action="store_true", default=False, help="Use RL agent to play. Otherwise, MPC is used.")
parser.add_argument("--tag", type=str, default=None, help="Tag for logging.")
parser.add_argument("--max_trials", type=int, default=1, help="Number of trials to run.")
parser.add_argument("--episode_length", type=float, default=20.0, help="Length of the episode in second.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from datetime import datetime
import gymnasium as gym
import os
import numpy as np
import torch

# rsl rl 
from rsl_rl.runners import OnPolicyRunner # master branch

# IsaacLab core
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

# Logger
from logger import BenchmarkLogger

ID = datetime.now().strftime("%Y-%m-%d_%H-%M")

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.episode_length_s = args_cli.episode_length
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint, ["model"])
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env.metadata["render_fps"] = 100/2

    # for logging
    if args_cli.use_rl:
        if args_cli.tag:
            name = f"play_rl/{args_cli.tag}"
        else:
            name = "play_rl"
    else:
        if args_cli.tag:
            name = f"play_mpc/{args_cli.tag}"
        else:
            name = "play_mpc"

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", name),
            # "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during inference.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env) # type: ignore

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device) # type: ignore
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )
    
    if args_cli.log:
        max_episode_length = int(env_cfg.episode_length_s/(env_cfg.decimation*env_cfg.sim.dt))
        logger = BenchmarkLogger(log_dir, name, num_envs=args_cli.num_envs, max_trials=args_cli.max_trials, max_episode_length=max_episode_length)
    max_trials = args_cli.max_trials
    episode_length_log = [0]*args_cli.num_envs
    episode_counter = [0]*args_cli.num_envs

    # reset environment
    obs, _ = env.get_observations()
    
    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            if args_cli.use_rl:
                actions = policy(obs)
            else:
                actions = torch.zeros(env.action_space.shape,dtype=torch.float32, device=args_cli.device) # type: ignore
            obs, _, dones, _ = env.step(actions)
            ppo_runner.alg.actor_critic.reset(dones) # reset hidden state of batch dim with done=1
        
        if args_cli.log:
            if args_cli.log_extra:
                rft=torch.cat([env.unwrapped.foot_depth.unsqueeze(2),  # type: ignore
                                    env.unwrapped.foot_angle_beta.unsqueeze(2),  # type: ignore
                                    env.unwrapped.foot_angle_gamma.unsqueeze(2),  # type: ignore
                                    env.unwrapped.foot_velocity, # type: ignore
                                    env.unwrapped.foot_accel, # type: ignore
                                    env.unwrapped.rft_force], dim=2) # type: ignore
                logger.log(state=env.unwrapped._state.cpu().numpy(), # type: ignore
                        obs=obs.cpu().numpy(),
                        raw_action=actions.cpu().numpy(),
                        action=env.unwrapped._actions_op.cpu().numpy(), # type: ignore
                        rft=rft.cpu().numpy(), 
                        done=dones.cpu().numpy(), 
                        )
            else:
                logger.log(state=env.unwrapped._state.cpu().numpy(), # type: ignore
                        obs=obs.cpu().numpy(),
                        raw_action=actions.cpu().numpy(),
                        action=env.unwrapped._actions_op.cpu().numpy(), # type: ignore
                        done=dones.cpu().numpy(), 
                        )
        # Incremenet episode length 
        for i  in range(args_cli.num_envs):
            episode_length_log[i] += 1
        
        # Convert done flag to numpy
        dones_np = dones.cpu().numpy()

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
