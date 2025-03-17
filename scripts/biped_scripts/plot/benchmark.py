import os
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

import numpy as np

# global

# rigid ground
CONTROLLERS = ["rl", "mpc"]
# CONTROLLERS = ["rl_accel_only"]
VTAGS = ["v0.15", "v0.2", "v0.25", "v0.3", "v0.35", "v0.4", "v0.45", "v0.5"]
MU_TAGS = ["mu_0.2", "mu_0.25"]

# # soft ground
# CONTROLLERS = ["rl", "mpc"]
# VTAGS = ["v0.1", "v0.15", "v0.2", "v0.25", "v0.3", "v0.35", "v0.4", "v0.45"]
# MU_TAGS = ["mu_0.3", "mu_0.3_compact_0.8"]


# observation indices
height_indies = 0
orientation_indies = slice(1, 5)
linear_velocity_indies = slice(5, 7)
angular_velocity_indies = 10
desired_linear_velocity_indies = slice(11, 13)
desired_angular_velocity_indies = 13
joint_pos_indies = slice(14, 24)
joint_vel_indies = slice(24, 34)
jont_effort_indices = slice(34, 44)
# previous_action_indies = slice(44, 50)
# mpc_action_indies = slice(50, 56)
# gait_contact_indices = slice(56, 58)
# gt_contact_indices = slice(58, 60)

def process_data(data_root:str, controller:str, v_tag:str, mu_tag:str):
    # data_root 
    # e.g ../../../../../logs/rsl_rl/ppo_rsl_rl_gru_friction_foot_placement/2025-02-04_20-02-06/logs
    data_root = os.path.join(data_root, f"play_{controller}", f"{v_tag}_{mu_tag}")
    
    state_dir = os.path.join(data_root, "state")
    obs_dir = os.path.join(data_root, "obs")
    action_dir = os.path.join(data_root, "action")
    episode_length_dir = os.path.join(data_root, "episode")
    
    # collect all the data
    state_files = glob.glob(os.path.join(state_dir, "*.pkl"))
    obs_files = glob.glob(os.path.join(obs_dir, "*.pkl"))
    action_files = glob.glob(os.path.join(action_dir, "*.pkl"))
    episode_files = glob.glob(os.path.join(episode_length_dir, "*.pkl"))
    
    # sort the files
    state_files.sort()
    obs_files.sort()
    action_files.sort()
    episode_files.sort()

    # load the data
    for i in range(len(state_files)):
        state_file = state_files[i]
        obs_file = obs_files[i]
        action_file = action_files[i]
        episode_file = episode_files[i]

        with open(state_file, "rb") as f:
            state = pickle.load(f)
        with open(obs_file, "rb") as f:
            obs = pickle.load(f)
        with open(action_file, "rb") as f:
            action = pickle.load(f)
        with open(episode_file, "rb") as f:
            episode_length_data = pickle.load(f)

    state_data = np.array(state)
    obs_data = np.array(obs)
    action_data = np.array(action)
    mpc_action_data = obs_data[:, :, :, 50:56]
    episode_length_data = np.array(episode_length_data)

    episode_length = state_data.shape[0]
    batch_size = state_data.shape[1]
    time_step = state_data.shape[2]

    state_data = state_data.reshape(episode_length*batch_size, time_step, -1)
    obs_data = obs_data.reshape(episode_length*batch_size, time_step, -1)
    action_data = action_data.reshape(episode_length*batch_size, time_step, -1)
    mpc_action_data = mpc_action_data.reshape(episode_length*batch_size, time_step, -1)
    episode_length_data = episode_length_data.reshape(-1)
    
    data_info = ""
    data_info += f"number of envs: {obs_data.shape[0]}\n"
    data_info += f"state dimension: {state_data.shape}\n"
    data_info += f"observersation dimension: {obs_data.shape}\n"
    data_info += f"action dimension: {action_data.shape}\n"
    print(data_info)
    
    return data_root, state_data, obs_data, action_data, mpc_action_data, episode_length_data

def save_benchmark_data(experiment_path:str):
    max_step = 1999
    max_length_s = 20
    
    for controller in CONTROLLERS:
        for v_tag in VTAGS:
            for mu_tag in MU_TAGS:
                data_root, state_data, obs_data, action_data, mpc_action_data, episode_length_data = process_data(experiment_path, controller, v_tag, mu_tag)
                
                velocity_x = obs_data[:, :, linear_velocity_indies][:, :, 0]
                desired_velocity_x = obs_data[:, :, desired_linear_velocity_indies][:, :, 0]

                ang_velocity = obs_data[:, :, angular_velocity_indies]
                desired_ang_velocity = obs_data[:, :, desired_angular_velocity_indies]

                step_length_rate = episode_length_data/max_step
                mean_step_length_rate = step_length_rate.mean()
                std_step_length_rate = step_length_rate.std()
                mean_step_length = max_length_s*mean_step_length_rate
                std_step_length = max_length_s*std_step_length_rate

                success_rate = episode_length_data == max_step
                succes_rate = np.sum(success_rate)/len(success_rate)
                if np.max(episode_length_data) == max_step:
                    alive_idx = np.squeeze(np.argwhere(episode_length_data==max_step))
                    vel_x_alive = velocity_x[alive_idx]
                    desired_vel_x_alive = desired_velocity_x[alive_idx]
                    ang_vel_alive = ang_velocity[alive_idx]
                    desired_ang_vel_alive = desired_ang_velocity[alive_idx]

                    if len(np.abs(vel_x_alive-desired_vel_x_alive).shape) ==1:
                        vel_x_alive = vel_x_alive[None, :]
                        desired_vel_x_alive = desired_vel_x_alive[None, :]
                        ang_vel_alive = ang_vel_alive[None, :]
                        desired_ang_vel_alive = desired_ang_vel_alive[None, :]
                    
                    vel_x_tracking_error = np.abs(vel_x_alive-desired_vel_x_alive)[:, :-1]
                    ang_vel_tracking_error = np.abs(ang_vel_alive-desired_ang_vel_alive)[:, :-1]

                    mean_vel_x_tracking_error = np.mean(vel_x_tracking_error)
                    std_vel_x_tracking_error = np.std(vel_x_tracking_error)
                    mean_ang_vel_tracking_error = np.mean(ang_vel_tracking_error)
                    std_ang_vel_tracking_error = np.std(ang_vel_tracking_error)
                else:
                    print("all trial failed")
                    mean_vel_x_tracking_error = 0
                    std_vel_x_tracking_error = 0
                    mean_ang_vel_tracking_error = 0
                    std_ang_vel_tracking_error = 0

                # score
                benchmark_score = {
                    "tag": f"{controller}_{v_tag}_{mu_tag}",
                    "mean_step_length_s": mean_step_length, 
                    "std_step_length_s": std_step_length,
                    "success_rate": succes_rate,
                    "mean_vel_x_tracking_error": mean_vel_x_tracking_error,
                    "std_vel_x_tracking_error": std_vel_x_tracking_error,
                    "mean_ang_vel_tracking_error": mean_ang_vel_tracking_error,
                    "std_ang_vel_tracking_error": std_ang_vel_tracking_error
                }
                
                message = ""
                message += f"tag: {controller}_{v_tag}_{mu_tag}\n"
                message += f"mean step length: {mean_step_length}\n"
                message += f"std step length: {std_step_length}\n"
                message += f"success rate: {succes_rate}\n"
                message += f"mean vel x tracking error: {mean_vel_x_tracking_error}\n"
                message += f"std vel x tracking error: {std_vel_x_tracking_error}\n"
                message += f"mean ang vel tracking error: {mean_ang_vel_tracking_error}\n"
                message += f"std ang vel tracking error: {std_ang_vel_tracking_error}\n"
                print(message)
                
                with open(os.path.join(data_root, "benchmark_score.pkl"), "wb") as f:
                    pickle.dump(benchmark_score, f)
    print("Done!!")

# # rigid ground #
PLOT_CONTROLLER = ["mpc", "rl_accel_only", "rl_fp_only", "rl"]
PLOT_CONTROLLER_NAMES = ["MPC", "RL-MPC-accel", "RL-MPC-fp", "RL-MPC-full"]
PLOT_VTAGS = {"v0.15":0.15, "v0.2":0.2, "v0.25":0.25, "v0.3":0.3, "v0.35":0.35, "v0.4":0.4, "v0.45":0.45, "v0.5":0.5}
PLOT_VTAGS_NAMES = [r"$v_x=0.15$", r"$v_x=0.2$", r"$v_x=0.25$", r"$v_x=0.3$", r"$v_x=0.35$", r"$v_x=0.4$", r"$v_x=0.45$", r"$v_x=0.5$"]
PLOT_MU_TAGS = {"mu_0.2":0.2, "mu_0.25":0.25}
PLOT_MU_TAGS_NAMES = [r"$\mu=0.2$", r"$\mu=0.25$"]

# # soft ground #
# PLOT_CONTROLLER = ["mpc", "rl"]
# PLOT_CONTROLLER_NAMES = ["MPC", "RL-MPC-full"]
# PLOT_VTAGS = {"v0.1":0.1, "v0.15":0.15, "v0.2":0.2, "v0.25":0.25, "v0.3":0.3, "v0.35":0.35, "v0.4":0.4, "v0.45":0.45}
# PLOT_VTAGS_NAMES = [r"$v_x=0.1$", r"$v_x=0.15$", r"$v_x=0.2$", r"$v_x=0.25$", r"$v_x=0.3$", r"$v_x=0.35$", r"$v_x=0.4$", r"$v_x=0.45$"]
# PLOT_MU_TAGS = {"mu_0.3":0.3, "mu_0.3_compact_0.8":0.2}
# PLOT_MU_TAGS_NAMES = [r"$\mu=0.3: stifness=\alpha_{CP}$", r"$\mu=0.3: stifness=0.8\alpha_{CP}$"]

def make_benchmark_plot(experiment_path:str):
    
    # store data
    scores_per_controller = []
    for controller in PLOT_CONTROLLER:
        scores_per_mu = []
        for mu_tag in PLOT_MU_TAGS.keys():
            scores_per_v = []
            for v_tag in PLOT_VTAGS.keys():
                data_root = os.path.join(experiment_path, f"play_{controller}", f"{v_tag}_{mu_tag}")
                benchmark_score_path = os.path.join(data_root, "benchmark_score.pkl")

                with open(benchmark_score_path, "rb") as f:
                    benchmark_score = pickle.load(f)
                mean_step_length_s = benchmark_score["mean_step_length_s"]
                std_step_length_s = benchmark_score["std_step_length_s"]
                # success_rate = benchmark_score["success_rate"]
                mean_vel_x_tracking_error = benchmark_score["mean_vel_x_tracking_error"]
                std_vel_x_tracking_error = benchmark_score["std_vel_x_tracking_error"]
                mean_ang_vel_tracking_error = benchmark_score["mean_ang_vel_tracking_error"]
                std_ang_vel_tracking_error = benchmark_score["std_ang_vel_tracking_error"]
                scores_per_v.append(
                    [[mean_step_length_s, std_step_length_s], 
                     [mean_vel_x_tracking_error, std_vel_x_tracking_error],
                     [mean_ang_vel_tracking_error, std_ang_vel_tracking_error]]
                )
            scores_per_mu.append(scores_per_v)
        scores_per_controller.append(scores_per_mu)
    
    scores_per_controller = np.array(scores_per_controller) # [controller, mu, v, score]
    bar_plot(scores_per_controller, PLOT_CONTROLLER_NAMES, PLOT_MU_TAGS_NAMES, PLOT_VTAGS_NAMES)

# scores 
# [0]: controller1 [[data_1], [data_2], ...]
# [1]: controller2 [[data_1], [data_2], ...]
# [2]: controller3 [[data_1], [data_2], ...]
def bar_plot(scores:np.ndarray, controller_tags:list, data_group_tags:list,  data_sub_group_tags:list):
    bar_width = 0.15
    num_matrics = 3
    num_data_groups = len(data_group_tags)
    matrix_groups = ["survival length (s)", r"$v_x^{error}$(m/s)", r"$\omega_z^{error}$(rad/s)"]
    # colors = ["red", "blue", "green"]
    fig, ax = plt.subplots(num_matrics, num_data_groups, figsize=(16, 8))
    
    for i in range(num_data_groups):
        ax[0, i].set_title(data_group_tags[i])
        for j in range(num_matrics):
            for k in range(len(controller_tags)):
                ax[j, i].bar(np.arange(len(data_sub_group_tags))+bar_width*k, scores[k, i, :, j, 0], bar_width, label=controller_tags[k])
                ax[j, i].errorbar(np.arange(len(data_sub_group_tags))+bar_width*k, scores[k, i, :, j, 0], yerr=scores[k, i, :, j, 1], fmt="o", color="black", capsize=3)
                ax[j, i].set_ylabel(matrix_groups[j])
                ax[j, i].set_xticks(np.arange(len(data_sub_group_tags))+bar_width/2)
                ax[j, i].set_xticklabels(data_sub_group_tags)
                if j == 1:
                    ax[j, i].set_ylim(0, 0.6)
                elif j == 2:
                    ax[j, i].set_ylim(0, 1.2)
                if i == 0 and j == 2:
                    ax[j, i].legend()
    plt.tight_layout()
    # plt.savefig("benchmark_rigid_slip_ground.pdf")
    # plt.savefig("benchmark_soft_terrain.pdf")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", type=str, default="../../../../../logs/rsl_rl/ppo_rsl_rl_gru_friction_foot_placement/2025-02-04_20-02-06/logs")
    args = parser.parse_args()
    
    # save_benchmark_data(args.experiment_path)
    make_benchmark_plot(args.experiment_path)