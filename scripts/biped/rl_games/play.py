# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--video_speed", type=float, default=1.0, help="Speed of the recorded video.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

parser.add_argument("--log", action="store_true", default=False, help="Log the environment.")
parser.add_argument("--episode_length", type=float, default=None, help="Length of the episode in second.")
parser.add_argument("--use_rl", action="store_true", default=False, help="Use RL agent to play. Otherwise, MPC is used.")
parser.add_argument("--max_trials", type=int, default=1, help="Number of trials to run.")
parser.add_argument("--perceptive", action="store_true", default=False, help="Use perceptive policy. Otherwise, blind policy is used.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import os
import time
import numpy as np
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

from logger import DictBenchmarkLogger

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Play with RL-Games agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    # env_cfg.seed = agent_cfg["params"]["seed"] # set seed from agent config
    if args_cli.episode_length is not None:
        env_cfg.episode_length_s = args_cli.episode_length
    env_cfg.inference = True

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # checkpoint_file="*.pth"
        # checkpoint_file = "last_manager_sac_rl_games_stepping_stone_mlp_blind_ep_10000_rew_4.855741.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # slow down recording speed
    env.metadata["render_fps"] = int((1/env.unwrapped.step_dt) * args_cli.video_speed)  # type: ignore

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
        
    # for logging
    name = "play_rl" if args_cli.use_rl else "play_mpc"

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", name), # type: ignore
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    if args_cli.log:
        max_episode_length = int(env_cfg.episode_length_s/(env_cfg.decimation*env_cfg.sim.dt))
        log_item = [
            'state', 'obs', 'raw_action', 'action', 'reward', 
            'grf', 'heightmap', 'global_pos', 'first_contact', 
            'terrain_out_of_bounds', 'bad_orientation','base_too_low', 'time_out'
            ]
        if args_cli.perceptive:
            log_item.append('ref_height')
        logger = DictBenchmarkLogger(
            log_dir, name, 
            num_envs=args_cli.num_envs, 
            max_trials=args_cli.max_trials, 
            max_episode_length=max_episode_length, 
            log_item=log_item,
            )
        
    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    # agent_cfg["params"]["load_checkpoint"] = True
    # agent_cfg["params"]["load_path"] = resume_path
    # print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()
    
    
    max_trials = args_cli.max_trials
    episode_length_log = [0]*args_cli.num_envs
    episode_counter = [0]*args_cli.num_envs

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    # convert obs to agent format
    obs = agent.obs_to_torch(obs)
    state = env.unwrapped.action_manager.get_term("mpc_action").state # type: ignore
    
    obs_prev = obs.clone()  # type: ignore
    state_prev = state.clone()  # type: ignore
    
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn: # type: ignore
        agent.init_rnn()
    
    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            if args_cli.use_rl:
                # print("obs: \n", obs)
                action = agent.get_action(obs, is_deterministic=agent.is_deterministic)
                # transformed_mean, mean, sigma = agent.get_action_distribution_params(obs) # type: ignore
                # print("tanh(mean):\n", transformed_mean[:, 7])
                # print("mean:\n", mean[:, 7])
                # print("sigma:\n", sigma)
            else:
                action = torch.zeros(env.unwrapped.action_space.shape, dtype=torch.float32, device=args_cli.device) # type: ignore
                # if args_cli.perceptive:
                #     action[:, 1] = -1.0 # perceptive policy
                # else:
                #     action[:, 7] = -1.0 # blind policy
            obs, _, dones, _ = env.step(action)
            obs = agent.obs_to_torch(obs)
            
            processed_actions = env.unwrapped.action_manager.get_term("mpc_action").processed_actions # type: ignore
            state = env.unwrapped.action_manager.get_term("mpc_action").state # type: ignore
            
            if args_cli.perceptive:
                # reward_items = ["undesired_contacts_toe"] # perceptive policy
                reward_items = ["undesired_contacts_toe", "foot_landing_penalty_left", "foot_landing_penalty_right"] # perceptive policy
            else:
                reward_items = ["track_lin_vel_xy_exp"] # blind policy
            reward_index = [env.unwrapped.reward_manager._term_names.index(item) for item in reward_items] # type: ignore
            reward = env.unwrapped.reward_manager._step_reward[:, reward_index] # type: ignore

            # extras
            grf = env.unwrapped.observation_manager._obs_buffer.get("force", None)  # type: ignore
            exteroception = env.unwrapped.observation_manager._obs_buffer.get("exteroception", None)  # type: ignore
            ref_height = env.unwrapped.action_manager.get_term("mpc_action").reference_height # type: ignore
            global_pos = env.unwrapped.observation_manager._obs_buffer.get("global_pos", None)  # type: ignore
            first_contact = env.unwrapped.observation_manager._obs_buffer.get("first_contact", None)  # type: ignore

            # get termination terms
            term_dones = env.unwrapped.termination_manager._term_dones
            terrain_out_of_bounds = term_dones.get("terrain_out_of_bounds", None)  # type: ignore
            bad_orientation = term_dones.get("bad_orientation", None)
            base_too_low = term_dones.get("base_too_low", None)  # type: ignore
            time_out = term_dones.get("time_out", None)  # type: ignore

            
            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None: # type: ignore
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        
        if args_cli.log:
            item_dict = {
                "state": state_prev.cpu().numpy(),  # type: ignore
                "obs": obs_prev.cpu().numpy(), # type: ignore
                "raw_action": action.cpu().numpy(),  # type: ignore
                "action": processed_actions.cpu().numpy(),
                "reward": reward.cpu().numpy(),  # type: ignore
                # "ref_height": ref_height,  # type: ignore
            }
            if grf is not None:
                item_dict["grf"] = grf.cpu().numpy()
            else:
                item_dict["grf"] = np.zeros((args_cli.num_envs, 6))

            if exteroception is not None:
                item_dict["heightmap"] = exteroception.cpu().numpy()
            else:
                item_dict["heightmap"] = np.zeros((args_cli.num_envs, 2)) 

            if args_cli.perceptive:
                item_dict["ref_height"] = ref_height
            
            if global_pos is not None:
                item_dict["global_pos"] = global_pos.cpu().numpy()
            else:
                item_dict["global_pos"] = np.zeros((args_cli.num_envs, 3))
            
            if first_contact is not None:
                item_dict["first_contact"] = first_contact.cpu().numpy()
            else:
                item_dict["first_contact"] = np.zeros((args_cli.num_envs, 2))

            if terrain_out_of_bounds is not None:
                item_dict["terrain_out_of_bounds"] = terrain_out_of_bounds.unsqueeze(1).float().cpu().numpy()
            else:
                item_dict["terrain_out_of_bounds"] = np.zeros((args_cli.num_envs, 1))

            if bad_orientation is not None:
                item_dict["bad_orientation"] = bad_orientation.unsqueeze(1).float().cpu().numpy()
            else:
                item_dict["bad_orientation"] = np.zeros((args_cli.num_envs, 1))

            if base_too_low is not None:
                item_dict["base_too_low"] = base_too_low.unsqueeze(1).float().cpu().numpy()
            else:
                item_dict["base_too_low"] = np.zeros((args_cli.num_envs, 1))

            if time_out is not None:
                item_dict["time_out"] = time_out.unsqueeze(1).float().cpu().numpy()
            else:
                item_dict["time_out"] = np.zeros((args_cli.num_envs, 1))

            logger.log(item_dict)
            
        # update buffer
        obs_prev = obs.clone()  # type: ignore
        state_prev = state.clone()  # type: ignore
        
        # Incremenet episode length 
        for i  in range(args_cli.num_envs):
            episode_length_log[i] += 1
        
        # Convert done flag to numpy
        dones_np = dones.cpu().numpy()
        # Check each agent's done flag
        for i, done_flag in enumerate(dones_np):
            if episode_counter[i] < max_trials: # only allow logging when episode counter is less than max trials to avoid memory overflow
                if done_flag == 1:
                    # print(f"[INFO] Env {i}: Episode {episode_counter[i]} completed with episode length {episode_length_log[i]}.")
                    if terrain_out_of_bounds[i]:
                        print(f"[INFO] Env {i}: Terrain out of bounds")
                    if bad_orientation[i]:
                        print(f"[INFO] Env {i}: Bad orientation")
                    if base_too_low[i]:
                        print(f"[INFO] Env {i}: Base too low")
                    if time_out[i]:
                        print(f"[INFO] Env {i}: Time out")

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
    # run the main function
    main()
    # close sim app
    simulation_app.close()