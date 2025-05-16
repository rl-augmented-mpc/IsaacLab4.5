import torch
from isaaclab.assets import Articulation
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix


class RobotCore:
    def __init__(self, articulation:Articulation, num_envs:int, foot_body_id:torch.Tensor)->None:
        self.articulation = articulation
        self.num_envs = num_envs
        self.foot_body_id = foot_body_id
        self.total_contact_point = len(foot_body_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_pos = torch.zeros((num_envs, 3), device=device)
        self._init_rot = torch.eye(3, device=device).unsqueeze(0).repeat(num_envs, 1, 1)
        
    # state reset
    def reset_default_pose(self, default_pose:torch.Tensor, env_id:torch.Tensor)->None:
        self._init_pos[env_id] = default_pose[:, :3]
        self._init_rot[env_id] = matrix_from_quat(default_pose[:, 3:7])
        
    def set_external_force(self, forces:torch.Tensor, body_id:torch.Tensor, env_id:torch.Tensor)->None:
        self.articulation.set_external_force_and_torque(forces, torch.zeros_like(forces), body_id, env_id) # type: ignore
    
    def set_external_torque(self, torques:torch.Tensor, body_id:torch.Tensor, env_id:torch.Tensor)->None:
        self.articulation.set_external_force_and_torque(torch.zeros_like(torques), torques, body_id, env_id) # type: ignore
    
    def set_external_force_and_torque(self, forces:torch.Tensor, torques:torch.Tensor, body_id:torch.Tensor, env_id:torch.Tensor)->None:
        self.articulation.set_external_force_and_torque(forces, torques, body_id, env_id) # type: ignore
    
    # motor control
    def set_joint_position_target(self, joint_action:torch.Tensor, joint_ids:torch.Tensor)->None:
        self.articulation.set_joint_position_target(joint_action, joint_ids)
    
    def set_joint_velocity_target(self, joint_action:torch.Tensor, joint_ids:torch.Tensor)->None:
        self.articulation.set_joint_velocity_target(joint_action, joint_ids)
    
    def set_joint_effort_target(self, joint_action:torch.Tensor, joint_ids:torch.Tensor)->None:
        self.articulation.set_joint_effort_target(joint_action, joint_ids) # type: ignore
    
    # hard-reset
    def write_root_pose_to_sim(self, root_pose:torch.Tensor, env_id:torch.Tensor)->None:
        self.articulation.write_root_pose_to_sim(root_pose, env_id) # type: ignore
    
    def write_root_velocity_to_sim(self, root_velocity:torch.Tensor, env_id:torch.Tensor)->None:
        self.articulation.write_root_velocity_to_sim(root_velocity, env_id) # type: ignore
    
    def write_joint_state_to_sim(self, joint_pos:torch.Tensor, joint_vel:torch.Tensor, joint_id:torch.Tensor, env_id:torch.Tensor)->None:
        self.articulation.write_joint_state_to_sim(joint_pos, joint_vel, joint_id, env_id) # type: ignore
    
    """
    default pose
    """
    
    @property
    def default_root_state(self)->torch.Tensor:
        return self.articulation.data.default_root_state
    
    @property
    def default_joint_pos(self)->torch.Tensor:
        return self.articulation.data.default_joint_pos
    
    @property
    def default_joint_vel(self)->torch.Tensor:
        return self.articulation.data.default_joint_vel
    
    """
    root state
    """
    
    @property
    def root_state_w(self)->torch.Tensor:
        return self.articulation.data.root_state_w
    
    @property
    def root_pos_w(self)->torch.Tensor:
        return self.articulation.data.root_pos_w
        # return self.articulation.data.root_com_pos_w
    
    @property
    def root_pos_local(self)->torch.Tensor:
        """
        root position wrt odom frame (i.e. initial frame)
        """
        root_pos_w = self.root_pos_w.clone()
        root_pos_w[:, :] -= self._init_pos[:, :]
        root_pos_w[:, 2] += self.default_root_state[:, 2]
        return torch.bmm(torch.transpose(self._init_rot, 1, 2), root_pos_w.view(-1, 3, 1)).view(-1, 3)
    
    @property
    def root_quat_w(self)->torch.Tensor:
        return self.articulation.data.root_quat_w
        # return self.articulation.data.root_com_quat_w
    
    @property
    def root_rot_mat_w(self)->torch.Tensor:
        return matrix_from_quat(self.root_quat_w)
    
    @property
    def root_quat_local(self)->torch.Tensor:
        """
        quaternion orientation wrt odom frame (i.e. initial frame)
        """
        rot_mat = matrix_from_quat(self.root_quat_w)
        root_rot_mat = torch.bmm(torch.transpose(self._init_rot, 1, 2), rot_mat) # find relative orientation from initial rotation
        return quat_from_matrix(root_rot_mat)
    
    @property
    def root_rot_mat_local(self)->torch.Tensor:
        """
        rotation matrix wrt odom frame (i.e. initial frame)
        """
        rot_mat = matrix_from_quat(self.root_quat_w)
        root_rot_mat = torch.bmm(torch.transpose(self._init_rot, 1, 2), rot_mat)
        return root_rot_mat
    
    @property
    def root_yaw_local(self)->torch.Tensor:
        """
        root yaw angle wrt odom frame (i.e. initial frame)
        """
        root_quat = self.root_quat_local.reshape(-1, 4)
        root_yaw = torch.atan2(2*(root_quat[:, 0]*root_quat[:, 3] + root_quat[:, 1]*root_quat[:, 2]), 1 - 2*(root_quat[:, 2]**2 + root_quat[:, 3]**2))
        root_yaw = torch.atan2(torch.sin(root_yaw), torch.cos(root_yaw))   
        return root_yaw
    
    @property
    def root_lin_vel_w(self)->torch.Tensor:
        return self.articulation.data.root_lin_vel_w
    
    @property
    def root_ang_vel_w(self)->torch.Tensor:
        return self.articulation.data.root_ang_vel_w
    
    @property
    def root_lin_vel_b(self)->torch.Tensor:
        # return self.articulation.data.root_lin_vel_b
        return self.articulation.data.root_com_lin_vel_b
    
    @property
    def root_ang_vel_b(self)->torch.Tensor:
        # return self.articulation.data.root_ang_vel_b
        return self.articulation.data.root_com_ang_vel_b
    
    """
    joint state
    """
    
    @property
    def joint_pos(self)->torch.Tensor:
        return self.articulation.data.joint_pos
    
    @property
    def joint_vel(self)->torch.Tensor:
        return self.articulation.data.joint_vel
    
    @property
    def joint_acc(self)->torch.Tensor:
        return self.articulation.data.joint_acc
    
    @property
    def joint_effort(self)->torch.Tensor:
        return self.articulation.data.applied_torque
        # return self.articulation.data.joint_torque
    
    @property
    def joint_pos_limit(self)->torch.Tensor:
        return self.articulation.data.joint_pos_limits

    @property   
    def joint_vel_limit(self)->torch.Tensor:
        return self.articulation.data.joint_vel_limits

    @property
    def joint_effort_limit(self)->torch.Tensor:
        return self.articulation.data.joint_effort_limits
    
    """
    body state
    """
    
    @property
    def body_pos_w(self)->torch.Tensor:
        return self.articulation.data.body_pos_w
    
    @property
    def body_quat_w(self)->torch.Tensor:
        return self.articulation.data.body_quat_w

    @property
    def body_lin_vel_w(self)->torch.Tensor:
        return self.articulation.data.body_lin_vel_w
    
    @property
    def body_ang_vel_w(self)->torch.Tensor:
        return self.articulation.data.body_ang_vel_w
    
    @property
    def body_lin_acc_w(self)->torch.Tensor:
        return self.articulation.data.body_lin_acc_w
    
    @property
    def body_jacobian(self)->torch.Tensor:
        """
        Return link jacobian wrt body frame
        assume body velocity occupies first 6 columns of jacobian,
        """
        jac = self.articulation.root_physx_view.get_jacobians()
        return jac[:, self.foot_body_id, :, 6:]
    
    @property
    def mass_matrix(self)->torch.Tensor:
        return self.articulation.root_physx_view.get_mass_matrices()
    
    """
    foot state
    """
    
    @property
    def foot_com_rot(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (N, 2, 3, 3)
        """
        foot_quat = self.body_quat_w[:, self.foot_body_id, :]
        foot_mat = matrix_from_quat(foot_quat.reshape(-1, 4))
        return foot_mat.reshape(-1, 2, 3, 3)
    
    @property
    def foot_com_rel_vel(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (N, 2, 3)
        """
        foot_rel_vel = self.body_lin_vel_w[:, self.foot_body_id, :] - self.root_lin_vel_w[:, None, :]
        return foot_rel_vel
    
    # TODO: rename this to foot_com_rel_vel
    @property
    def foot_com_vel(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (N, 2, 3)
        """
        foot_vel = self.body_lin_vel_w[:, self.foot_body_id, :]
        return foot_vel
    
    @property
    def foot_com_ang_vel(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (N, 2, 3)
        """
        foot_ang_vel = self.body_ang_vel_w[:, self.foot_body_id, :]
        return foot_ang_vel
    
    @property
    def foot_com_vel_b(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (N, 2, 3)
        """
        foot_vel = self.foot_com_vel.clone()
        foot_vel = torch.bmm(self.foot_com_rot.reshape(-1, 3, 3).transpose(1,2), foot_vel.reshape(-1, 3, 1)).view(-1, 2, 3)
        return foot_vel
    
    @property
    def foot_com_ang_vel_b(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (N, 2, 3)
        """
        foot_ang_vel = self.foot_com_ang_vel.clone()
        foot_ang_vel = torch.bmm(self.foot_com_rot.reshape(-1, 3, 3).transpose(1,2), foot_ang_vel.reshape(-1, 3, 1)).view(-1, 2, 3)
        return foot_ang_vel
    
    
    @property
    def foot_pos(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (num_envs, num_contact_points, 3)
        """
        foot_pos = self.body_pos_w[:, self.foot_body_id, :]
        foot_pos[:, :, 2] =- 0.04
        return foot_pos
    
    @property
    def foot_pos_local(self)->torch.Tensor:
        """
        foot position in odom frame
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (num_envs, num_contact_points, 3, 3)
        """
        foot_pos = self.foot_pos.clone()
        foot_pos[:, :, :2] -= self._init_pos[:, None, :2]
        rot_mat = self._init_rot.clone().unsqueeze(1).repeat(1, self.total_contact_point, 1, 1).view(-1, 3, 3)
        return torch.bmm(rot_mat.transpose(1,2), foot_pos.view(-1, 3, 1)).view(-1, self.total_contact_point, 3)
    
    @property
    def foot_pos_b(self)->torch.Tensor:
        """
        foot position in body frame
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (num_envs, num_contact_points, 3, 3)
        """
        foot_pos = self.foot_pos.clone()
        foot_pos[:, :, :2] -= self.root_pos_w[:, None, :2]
        rot_mat = self.root_rot_mat_w.clone().unsqueeze(1).repeat(1, self.total_contact_point, 1, 1).view(-1, 3, 3)
        return torch.bmm(rot_mat.transpose(1,2), foot_pos.view(-1, 3, 1)).view(-1, self.total_contact_point, 3)
    
    @property
    def foot_quat(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (N, num_contact_points, 4)
        """
        foot_quat = self.body_quat_w[:, self.foot_body_id, :]
        return foot_quat
    
    @property
    def foot_rot_mat(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (N, num_contact_points, 3, 3)
        """
        foot_quat = self.body_quat_w[:, self.foot_body_id, :]
        foot_mat = matrix_from_quat(foot_quat.reshape(-1, 4))
        return foot_mat.reshape(-1, self.total_contact_point, 3, 3)
    
    @property
    def foot_vel(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (N, num_contact_points, 3)
        """
        foot_vel = self.body_lin_vel_w[:, self.foot_body_id, :]
        return foot_vel
    
    # @property
    # def foot_vel(self)->torch.Tensor:
    #     """
    #     Compute foot patch velocity using kinematic equation
    #     """
    #     ankle_lin_vel = self.body_lin_vel_w[:, -self.foot_patch_num-2:-self.foot_patch_num, :]
    #     ankle_ang_vel = self.body_ang_vel_w[:, -self.foot_patch_num-2:-self.foot_patch_num, :]
    #     foot_rel_pos_left = self.body_pos_w[:, -self.foot_patch_num:-self.foot_patch_num//2, :] - self.body_pos_w[:, -self.foot_patch_num-2:-self.foot_patch_num-1, :]
    #     foot_rel_pos_right = self.body_pos_w[:, -self.foot_patch_num//2:, :] - self.body_pos_w[:, -self.foot_patch_num-1:-self.foot_patch_num, :]
    #     foot_vel_left = ankle_lin_vel[:, 0:1, :] + torch.cross(ankle_ang_vel[:, 0:1, :], foot_rel_pos_left, dim=2) # (num_envs, left_contact_points, 3)
    #     foot_vel_right = ankle_lin_vel[:, 1:2, :] + torch.cross(ankle_ang_vel[:, 1:2, :], foot_rel_pos_right, dim=2) # (num_envs, right_contact_points, 3)
    #     return torch.cat((foot_vel_left, foot_vel_right), dim=1)
    
    @property
    def foot_vel_local(self)->torch.Tensor:
        """
        foot velocity in odom frame
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (num_envs, num_contact_points, 3, 3)
        """
        foot_vel = self.foot_vel.clone()
        rot_mat = self._init_rot.clone().unsqueeze(1).repeat(1, self.total_contact_point, 1, 1).view(-1, 3, 3)
        foot_vel = torch.bmm(rot_mat.transpose(1, 2), foot_vel.view(-1, 3, 1)).view(-1, self.total_contact_point, 3)
        return foot_vel
    
    @property
    def foot_vel_b(self)->torch.Tensor:
        """
        foot velocity in body frame
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (num_envs, num_contact_points, 3, 3)
        """
        foot_vel = self.foot_vel.clone()
        rot_mat = self.root_rot_mat_w.clone().unsqueeze(1).repeat(1, self.total_contact_point, 1, 1).view(-1, 3, 3)
        foot_vel = torch.bmm(rot_mat.transpose(1,2), foot_vel.view(-1, 3, 1)).view(-1, self.total_contact_point, 3)
        return foot_vel
    
    @property
    def foot_ang_vel_b(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (num_env, num_contact_points, 3)
        """
        foot_ang_vel = self.body_ang_vel_w[:, self.foot_body_id, :]
        foot_ang_vel = torch.bmm(self.foot_rot_mat.reshape(-1, 3, 3).transpose(1,2), foot_ang_vel.reshape(-1, 3, 1)).squeeze(-1).reshape(-1, self.total_contact_point, 3)
        return foot_ang_vel
    
    @property
    def foot_accel(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (N, num_contact_points, 3)
        """
        foot_vel = self.body_lin_acc_w[:, self.foot_body_id, :]
        return foot_vel
    
    @property
    def foot_pos_z(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (N, num_contact_points)
        """
        foot_pos = self.body_pos_w[:, self.foot_body_id, :]
        return foot_pos[:, :, -1]
    
    ## == Euler angles in ZYX order == ##
    @property
    def foot_roll_angle(self)->torch.Tensor:
        """
        Returns:
            beta (torch.Tensor): Foot angle of attack in radian (num_env, num_contact_points)
        """
        foot_quat = self.foot_quat.reshape(-1, 4)
        roll = torch.atan2(2*(foot_quat[:, 0]*foot_quat[:, 1] + foot_quat[:, 2]*foot_quat[:, 3]), 1-2*(foot_quat[:, 1]**2 + foot_quat[:, 2]**2))
        roll = torch.atan2(torch.sin(roll), torch.cos(roll))
        return roll.view(-1, self.total_contact_point)
    
    
    @property
    def foot_pitch_angle(self)->torch.Tensor:
        """
        Returns:
            beta (torch.Tensor): Foot angle of attack in radian (num_env, num_contact_points)
        """
        foot_quat = self.foot_quat.reshape(-1, 4)
        arg_asin = 2*(foot_quat[:, 0]*foot_quat[:, 2] - foot_quat[:, 3]*foot_quat[:, 1])
        arg_asin = torch.clamp(arg_asin, -1, 1) # avoid singularity
        pitch = torch.asin(arg_asin)
        pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch)) #standardize to -pi to pi
        return pitch.view(-1, self.total_contact_point)
    
    @property
    def foot_yaw_angle(self)->torch.Tensor:
        """
        Returns:
            beta (torch.Tensor): Foot angle of attack in radian (num_env, num_contact_points)
        """
        foot_quat = self.foot_quat.reshape(-1, 4)
        yaw = torch.atan2(2*(foot_quat[:, 0]*foot_quat[:, 3] + foot_quat[:, 1]*foot_quat[:, 2]), 1-2*(foot_quat[:, 2]**2 + foot_quat[:, 3]**2))
        yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw)) #standardize to -pi to pi
        return yaw.view(-1, self.total_contact_point)
    
    @property
    def foot_roll_yaw_matrix(self) -> torch.Tensor:
        """
        transformation from foot local frame to world frame
        Returns:
            foot_roll_yaw_matrix (torch.Tensor): Rotation matrices computed from foot roll and yaw angles,
                                                with pitch fixed to zero.
                                                Shape: (N, total_contact_point, 3, 3)
        """
        roll = self.foot_roll_angle  # Rotation about the X-axis.
        yaw = self.foot_yaw_angle    # Rotation about the Z-axis.
        
        # Compute cosine and sine values for roll and yaw.
        c_r = torch.cos(roll)
        s_r = torch.sin(roll)
        c_y = torch.cos(yaw)
        s_y = torch.sin(yaw)
        
        # Create helper tensors (with shape matching the batched angles).
        ones = torch.ones_like(roll)
        zeros = torch.zeros_like(roll)
        
        R_x = torch.stack([
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, c_r, -s_r], dim=-1),
            torch.stack([zeros, s_r, c_r], dim=-1)
        ], dim=-2)  # R_x has shape (N, total_contact_point, 3, 3)
        
        R_z = torch.stack([
            torch.stack([c_y, -s_y, zeros], dim=-1),
            torch.stack([s_y, c_y, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1)
        ], dim=-2)  # R_z has shape (N, total_contact_point, 3, 3)
        
        # yaw-pitch-roll order
        # transformation matrix is R = Rx*Rz
        # this function gives rotation matrix -> rot = R.T = Rz.T*Rx.T where Rz.T = Rz and Rx.T = Rx
        foot_mat = R_z @ R_x
        
        return foot_mat
    
    @property
    def foot_beta_angle(self)->torch.Tensor:
        """
        Returns:
            beta (torch.Tensor): Foot angle of attack in radian (N, num_contact_points)
        """
        beta = -self.foot_pitch_angle
        return beta.view(-1, self.total_contact_point)
    
    # gamma retrieved from velocity wrt local coordinate (where roll and yaw are aligned with body frame)
    @property
    def foot_gamma_angle(self)->torch.Tensor:
        """
        Commpute gamma using linear velocity wrt body frame (roll, yaw aligned with body frame) of each foot links
        Returns:
            gamma (torch.Tensor): Foot intrusion angle in radian (N, num_contact_points)
        """
        foot_vel = self.body_lin_vel_w[:, self.foot_body_id, :].reshape(-1, 3)
        foot_vel = torch.bmm(self.foot_roll_yaw_matrix.reshape(-1, 3, 3).transpose(1,2), foot_vel[:, :, None]).squeeze(-1) # get velocity wrt local coordinate (pitch is 0)
        gamma = torch.atan2(-foot_vel[:, 2], foot_vel[:, 0])
        gamma = torch.atan2(torch.sin(gamma), torch.cos(gamma)) #standardize to -pi to pi
        
        # map gamma (-pi, pi) to (-pi/2, pi/2) due to RFT convention
        sign_mask = gamma > 0
        coordinate_mask = torch.abs(gamma) < torch.pi/2
        gamma[sign_mask*~coordinate_mask] = torch.pi - gamma[sign_mask*~coordinate_mask]
        gamma[~sign_mask*~coordinate_mask] = -torch.pi - gamma[~sign_mask*~coordinate_mask]
        
        return gamma.view(-1, self.total_contact_point)
    
    @property
    def body_names(self)->list:
        return self.articulation.body_names