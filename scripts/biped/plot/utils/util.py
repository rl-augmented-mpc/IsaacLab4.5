import os
from glob import glob
import pickle
import warnings
import numpy as np

def quaternion_to_euler(quats):
    """
    Convert a batch of quaternions to Euler angles (roll, pitch, yaw) using only NumPy.
    
    Parameters:
        quats (numpy.ndarray): Shape (batch_size, 4), where each row is [x, y, z, w]
    
    Returns:
        numpy.ndarray: Shape (batch_size, 3), where each row is [roll, pitch, yaw] in radians
    """
    w = quats[:, 0]
    x = quats[:, 1]
    y = quats[:, 2]
    z = quats[:, 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))  # Handle gimbal lock

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    # normalize to -pi to pi
    roll = np.arctan2(np.sin(roll), np.cos(roll))
    pitch = np.arctan2(np.sin(pitch), np.cos(pitch))
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))

    return np.stack([roll, pitch, yaw], axis=1)  # Shape (batch_size, 3)

def quaternion_to_rotation_matrix(q):
    """
    Convert one or more unit quaternions into rotation matrices.
    
    Parameters
    ----------
    q : array-like, shape (..., 4)
        Quaternions in [w, x, y, z] format. The “...” can be any batch shape,
        e.g. (N,4) for N quaternions or just (4,) for a single quaternion.
    
    Returns
    -------
    R : ndarray, shape (..., 3, 3)
        Rotation matrices corresponding to each quaternion.
    """
    q = np.asarray(q, dtype=float)
    q[q[:, 0]<1e-4, 0] = 1.0  # avoid division by zero
    if q.ndim == 1:
        # single quaternion -> treat as batch size 1
        q = q[np.newaxis, :]
    assert q.shape[-1] == 4, "Last dimension must be 4"
    
    # flatten batch dims
    flat_q = q.reshape(-1, 4)
    
    # normalize (in case inputs aren't exactly unit-length)
    norms = np.linalg.norm(flat_q, axis=1, keepdims=True)
    flat_q = flat_q / norms
    
    w, x, y, z = flat_q.T
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    R_flat = np.empty((flat_q.shape[0], 3, 3))
    R_flat[:, 0, 0] = 1 - 2*(yy + zz)
    R_flat[:, 0, 1] =     2*(xy - wz)
    R_flat[:, 0, 2] =     2*(xz + wy)
    R_flat[:, 1, 0] =     2*(xy + wz)
    R_flat[:, 1, 1] = 1 - 2*(xx + zz)
    R_flat[:, 1, 2] =     2*(yz - wx)
    R_flat[:, 2, 0] =     2*(xz - wy)
    R_flat[:, 2, 1] =     2*(yz + wx)
    R_flat[:, 2, 2] = 1 - 2*(xx + yy)
    
    # reshape back to original batch shape
    R = R_flat.reshape(q.shape[:-1] + (3, 3))
    
    # if input was a single quaternion, squeeze out the batch dim
    return R[0] if R.shape[:-2] == () else R

def cluster_time_points(t_array, threshold=0.01):
    """
    Groups time points into chunks if they are within `threshold` seconds apart.
    Returns a list of (start_time, end_time) for each chunk.
    """
    if len(t_array) == 0:
        return []

    t_array = np.sort(t_array)
    chunks = []
    start = t_array[0]
    end = t_array[0]

    for t in t_array[1:]:
        if t - end <= threshold:
            end = t  # extend current chunk
        else:
            chunks.append((start, end))
            start = t
            end = t
    chunks.append((start, end))  # don't forget the last one
    return chunks

def process_data(data_dir:str):
    state_dir = os.path.join(data_dir, "state")
    obs_dir = os.path.join(data_dir, "obs")
    action_dir = os.path.join(data_dir, "action")
    episode_length_dir = os.path.join(data_dir, "episode")
    reward_dir = os.path.join(data_dir, "reward")


    # collect all the data
    state_files = glob(os.path.join(state_dir, "*.pkl"))
    obs_files = glob(os.path.join(obs_dir, "*.pkl"))
    action_files = glob(os.path.join(action_dir, "*.pkl"))
    episode_files = glob(os.path.join(episode_length_dir, "*.pkl"))
    reward_files = glob(os.path.join(reward_dir, "*.pkl"))


    # sort the files
    state_files.sort()
    obs_files.sort()
    action_files.sort()
    episode_files.sort()
    reward_files.sort()

    # load the data
    for i in range(len(state_files)):
        state_file = state_files[i]
        obs_file = obs_files[i]
        action_file = action_files[i]
        episode_file = episode_files[i]
        reward_file = reward_files[i]

        with open(state_file, "rb") as f:
            state = pickle.load(f)
        with open(obs_file, "rb") as f:
            obs = pickle.load(f)
        with open(action_file, "rb") as f:
            action = pickle.load(f)
        with open(episode_file, "rb") as f:
            episode_length_data = pickle.load(f)
        with open(reward_file, "rb") as f:
            reward = pickle.load(f)

    state_data = np.array(state)
    obs_data = np.array(obs)
    action_data = np.array(action)
    mpc_action_data = obs_data[:, :, :, 50:56]
    episode_length_data = np.array(episode_length_data)
    reward_data = np.array(reward)

    num_trials = state_data.shape[0]
    batch_size = state_data.shape[1]
    time_step = state_data.shape[2]

    state_data = state_data.reshape(num_trials*batch_size, time_step, -1)
    obs_data = obs_data.reshape(num_trials*batch_size, time_step, -1)
    action_data = action_data.reshape(num_trials*batch_size, time_step, -1)
    mpc_action_data = mpc_action_data.reshape(num_trials*batch_size, time_step, -1)
    episode_length_data = episode_length_data.reshape(-1)
    reward_data = reward_data.reshape(num_trials*batch_size, time_step, -1)
    
    return (
        state_data, 
        obs_data,
        action_data,
        reward_data,
        mpc_action_data, 
        episode_length_data, 
    )

def process_data_with_height(data_dir:str):
    state_dir = os.path.join(data_dir, "state")
    obs_dir = os.path.join(data_dir, "obs")
    action_dir = os.path.join(data_dir, "action")
    episode_length_dir = os.path.join(data_dir, "episode")
    reward_dir = os.path.join(data_dir, "reward")


    # collect all the data
    state_files = glob(os.path.join(state_dir, "*.pkl"))
    obs_files = glob(os.path.join(obs_dir, "*.pkl"))
    action_files = glob(os.path.join(action_dir, "*.pkl"))
    episode_files = glob(os.path.join(episode_length_dir, "*.pkl"))
    reward_files = glob(os.path.join(reward_dir, "*.pkl"))


    # sort the files
    state_files.sort()
    obs_files.sort()
    action_files.sort()
    episode_files.sort()
    reward_files.sort()

    # load the data
    for i in range(len(state_files)):
        state_file = state_files[i]
        obs_file = obs_files[i]
        action_file = action_files[i]
        episode_file = episode_files[i]
        reward_file = reward_files[i]

        with open(state_file, "rb") as f:
            state = pickle.load(f)
        with open(obs_file, "rb") as f:
            obs = pickle.load(f)
        with open(action_file, "rb") as f:
            action = pickle.load(f)
        with open(episode_file, "rb") as f:
            episode_length_data = pickle.load(f)
        with open(reward_file, "rb") as f:
            reward = pickle.load(f)

    state_data = np.array(state)
    obs_data = np.array(obs)
    action_data = np.array(action)
    mpc_action_data = obs_data[:, :, :, 50:56]
    episode_length_data = np.array(episode_length_data)
    reward_data = np.array(reward)

    num_trials = state_data.shape[0]
    batch_size = state_data.shape[1]
    time_step = state_data.shape[2]

    state_data = state_data.reshape(num_trials*batch_size, time_step, -1)

    # process height data
    num_history = 1
    height_w, height_h = 21, 21 # resolution 0.05
    height_scan_num = height_w * height_h
    obs_data = obs_data.reshape(num_trials*batch_size, time_step, -1)
    height_data = obs_data[:, :, -height_scan_num:]
    
    obs_data = obs_data[:, :, :-height_scan_num].reshape(num_trials*batch_size, time_step, num_history, -1)[:, :, -1, :]
    action_data = action_data.reshape(num_trials*batch_size, time_step, -1)
    mpc_action_data = mpc_action_data.reshape(num_trials*batch_size, time_step, -1)
    episode_length_data = episode_length_data.reshape(-1)
    reward_data = reward_data.reshape(num_trials*batch_size, time_step, -1)
    
    return (
        state_data, 
        obs_data,
        action_data,
        reward_data,
        mpc_action_data, 
        height_data, 
        episode_length_data, 
    )


def load_processed_data(data_root, dt_policy=0.01):
    """
    Load and process data from a given root directory.
    Returns a dictionary with standardized keys, prefixed by `label` (e.g., 'rl', 'mpc').
    """
    state_data, obs_data, action_data, reward_data, mpc_action_data, episode_length_data = process_data(data_root)

    num_envs = obs_data.shape[0]

    # data indices
    height_indies = 0
    linear_velocity_indies = slice(1, 4)
    angular_velocity_indies = 6
    orientation_indies = slice(7, 10)
    desired_linear_velocity_indies = slice(10, 12)
    desired_angular_velocity_indies = 12
    joint_pos_indies = slice(13, 23)
    joint_vel_indies = slice(23, 33)
    jont_effort_indices = slice(33, 43)
    swing_phase_indices = slice(43, 45)
    foot_placement_b_indices = slice(45, 49)
    foot_position_b_indices = slice(49, 55)
    reference_foot_position_b_indices = slice(55, 61)

    # state indices
    state_position_indices = slice(0, 3)
    state_quat_indices = slice(3, 7)
    velocity_indices = slice(7, 10)
    ang_velocity_indices = slice(10, 13)


    # state indices
    state_position_indices = slice(0, 3)
    state_quat_indices = slice(3, 7)
    velocity_indices = slice(7, 10)
    ang_velocity_indices = slice(10, 13)

    # --- Process as before ---
    velocity = obs_data[:, :, linear_velocity_indies]
    velocity_x = velocity[:, :, 0]
    desired_velocity = obs_data[:, :, desired_linear_velocity_indies]
    desired_velocity_x = desired_velocity[:, :, 0]
    ang_velocity = obs_data[:, :, angular_velocity_indies]
    desired_ang_velocity = obs_data[:, :, desired_angular_velocity_indies]

    joint_pos = obs_data[:, :, joint_pos_indies]
    joint_vel = obs_data[:, :, joint_vel_indies]
    joint_effort = obs_data[:, :, jont_effort_indices]

    swing_phase = obs_data[:, :, swing_phase_indices]
    foot_placement_b = obs_data[:, :, foot_placement_b_indices]
    foot_position_b = obs_data[:, :, foot_position_b_indices].reshape(num_envs, -1, 2, 3)
    reference_foot_position_b = obs_data[:, :, reference_foot_position_b_indices].reshape(num_envs, -1, 2, 3)

    # State
    position = state_data[:, :, state_position_indices]
    quat = state_data[:, :, state_quat_indices]
    orientation = (180/np.pi) * quaternion_to_euler(quat.reshape(-1, 4)).reshape(num_envs, -1, 3)
    velocity = state_data[:, :, velocity_indices]
    ang_velocity = state_data[:, :, ang_velocity_indices]

    position_des = np.zeros_like(position)
    orientation_des = np.zeros_like(position)

    orientation_des[:, :, 2] = np.cumsum(desired_ang_velocity*dt_policy, axis=1)
    orientation_des[:, :, 2] = np.arctan2(np.sin(orientation_des[:, :, 2]), np.cos(orientation_des[:, :, 2]))

    position_des[:, :, 0] = np.cumsum(desired_velocity_x*np.cos(orientation_des[:, :, 2])*dt_policy, axis=1)
    position_des[:, :, 1] = np.cumsum(desired_velocity_x*np.sin(orientation_des[:, :, 2])*dt_policy, axis=1)
    position_des[:, :, 2] = 0.55 * np.ones_like(position_des[:, :, 2])

    orientation_des = (180/np.pi) * orientation_des

    rot = quaternion_to_rotation_matrix(quat.reshape(-1, 4)).reshape(num_envs, -1, 3, 3)
    velocity_w = (rot @ velocity.reshape(num_envs, -1, 3, 1)).reshape(num_envs, -1, 3)
    ang_velocity_w = (rot @ ang_velocity.reshape(num_envs, -1, 3, 1)).reshape(num_envs, -1, 3)

    foot_position_w = np.zeros_like(foot_position_b)
    reference_foot_position_w = np.zeros_like(reference_foot_position_b)

    foot_position_w[:, :, 0, :] = ((rot.reshape(-1, 3, 3) @ foot_position_b[:, :, 0, :].reshape(-1, 3, 1)).reshape(num_envs, -1, 3)) + position
    foot_position_w[:, :, 1, :] = ((rot.reshape(-1, 3, 3) @ foot_position_b[:, :, 1, :].reshape(-1, 3, 1)).reshape(num_envs, -1, 3)) + position
    reference_foot_position_w[:, :, 0, :] = ((rot.reshape(-1, 3, 3) @ reference_foot_position_b[:, :, 0, :].reshape(-1, 3, 1)).reshape(num_envs, -1, 3)) + position
    reference_foot_position_w[:, :, 1, :] = ((rot.reshape(-1, 3, 3) @ reference_foot_position_b[:, :, 1, :].reshape(-1, 3, 1)).reshape(num_envs, -1, 3)) + position

    mask = swing_phase == 0
    mask[:, 0, 1] = False
    stance_foot_position_w = (foot_position_w * mask[:, :, :, None]).sum(axis=2)
    stance_foot_position_b = (foot_position_b.reshape(num_envs, -1, 2, 3) * mask[:, :, :, None]).sum(axis=2)

    body_foot_angle = np.arctan2(np.abs(stance_foot_position_b[:, :, 0]), np.abs(stance_foot_position_b[:, :, 2]))
    foot_lateral_distance = foot_position_b[:, :, 0, 1] - foot_position_b[:, :, 1, 1]


    # extras
    with open(os.path.join(data_root, "grw/grw.pkl"), "rb") as f:
        grw = pickle.load(f)
    grw = np.array(grw)
    grw = grw.reshape(grw.shape[0]*grw.shape[1], -1, 12)

    return {
        "position": position,
        "orientation": orientation,
        "velocity": velocity,
        "ang_velocity": ang_velocity,
        "desired_position": position_des,
        "desired_orientation": orientation_des,
        "desired_velocity": desired_velocity,
        "desired_ang_velocity": desired_ang_velocity,
        "foot_position_w": foot_position_w,
        "foot_position_b": foot_position_b,
        "reference_foot_position_w": reference_foot_position_w,
        "reference_foot_position_b": reference_foot_position_b,
        "body_foot_angle": body_foot_angle,
        "foot_lateral_distance": foot_lateral_distance,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "joint_effort": joint_effort,
        "swing_phase": swing_phase,
        "foot_placement_b": foot_placement_b,
        "grw": grw,
        "action": action_data,
        "mpc_action": mpc_action_data,
        "episode_length": episode_length_data,
    }