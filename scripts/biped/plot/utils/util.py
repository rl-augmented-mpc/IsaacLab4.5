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