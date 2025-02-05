import torch
from isaaclab.assets import Articulation
from isaaclab.utils.math import matrix_from_quat


class RobotCore:
    def __init__(self, articulation:Articulation, foot_patch_num:int=8)->None:
        self.articulation = articulation
        self.foot_patch_num = foot_patch_num
        
    # state reset
    def reset_default_pose(self, default_pose:torch.Tensor, env_id:torch.Tensor)->None:
        self.articulation.data.default_root_state[env_id, :7] = default_pose
        
    def set_external_force(self, forces:torch.Tensor, body_id:torch.Tensor, env_id:torch.Tensor)->None:
        self.articulation.set_external_force_and_torque(forces, torch.zeros_like(forces), body_id, env_id)
    
    def set_external_torque(self, torques:torch.Tensor, body_id:torch.Tensor, env_id:torch.Tensor)->None:
        self.articulation.set_external_force_and_torque(torch.zeros_like(torques), torques, body_id, env_id)
    
    def set_external_wrench(self, forces:torch.Tensor, torques:torch.Tensor, body_id:torch.Tensor, env_id:torch.Tensor)->None:
        self.articulation.set_external_force_and_torque(forces, torques, body_id, env_id)
    
    # motor control
    def set_joint_effort_target(self, joint_action:torch.Tensor, joint_ids:torch.Tensor)->None:
        self.articulation.set_joint_effort_target(joint_action, joint_ids)
    
    # hard-reset
    def write_root_pose_to_sim(self, root_pose:torch.Tensor, env_id:torch.Tensor)->None:
        self.articulation.write_root_pose_to_sim(root_pose, env_id)
    
    def write_root_velocity_to_sim(self, root_velocity:torch.Tensor, env_id:torch.Tensor)->None:
        self.articulation.write_root_velocity_to_sim(root_velocity, env_id)
    
    def write_joint_state_to_sim(self, joint_pos:torch.Tensor, joint_vel:torch.Tensor, joint_id:torch.Tensor, env_id:torch.Tensor)->None:
        self.articulation.write_joint_state_to_sim(joint_pos, joint_vel, joint_id, env_id)
    
    @property
    def default_root_state(self)->torch.Tensor:
        return self.articulation.data.default_root_state
    
    @property
    def default_joint_pos(self)->torch.Tensor:
        return self.articulation.data.default_joint_pos
    
    @property
    def default_joint_vel(self)->torch.Tensor:
        return self.articulation.data.default_joint_vel
    
    @property
    def root_pos_w(self)->torch.Tensor:
        return self.articulation.data.root_pos_w
    
    @property
    def root_quat_w(self)->torch.Tensor:
        return self.articulation.data.root_quat_w
    
    @property
    def root_lin_vel_w(self)->torch.Tensor:
        return self.articulation.data.root_lin_vel_w
    
    @property
    def root_ang_vel_w(self)->torch.Tensor:
        return self.articulation.data.root_ang_vel_w
    
    @property
    def root_lin_vel_b(self)->torch.Tensor:
        return self.articulation.data.root_lin_vel_b
    
    @property
    def root_ang_vel_b(self)->torch.Tensor:
        return self.articulation.data.root_ang_vel_b
    
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
        return self.articulation.data.joint_effort
    
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
    def foot_pos(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (num_env, num_contact_points, 3)
        """
        foot_pos = self.body_pos_w[:, -self.foot_patch_num:, :]
        return foot_pos
    
    @property
    def foot_quat(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (num_env, num_contact_points, 3)
        """
        foot_quat = self.body_quat_w[:, -self.foot_patch_num:, :]
        return foot_quat
    
    @property
    def foot_rot_mat(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (num_env, num_contact_points, 3, 3)
        """
        foot_quat = self.body_quat_w[:, -self.foot_patch_num:, :]
        foot_mat = matrix_from_quat(foot_quat.reshape(-1, 4))
        return foot_mat.reshape(-1, self.foot_patch_num, 3, 3)
        
    
    @property
    def foot_vel(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (num_env, num_contact_points, 3)
        """
        foot_vel = self.body_lin_vel_w[:, -self.foot_patch_num:, :]
        return foot_vel
    
    @property
    def foot_ang_vel_b(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (num_env, num_contact_points, 3)
        """
        foot_ang_vel = self.body_ang_vel_w[:, -self.foot_patch_num:, :]
        foot_ang_vel = torch.bmm(self.foot_rot_mat.reshape(-1, 3, 3).transpose(1,2), foot_ang_vel.reshape(-1, 3, 1)).squeeze(-1).reshape(-1, self.foot_patch_num, 3)
        return foot_ang_vel
    
    @property
    def foot_accel(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (num_env, num_contact_points, 3)
        """
        foot_vel = self.body_lin_acc_w[:, -self.foot_patch_num:, :]
        return foot_vel
    
    @property
    def foot_depth(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (num_env, num_contact_points)
        """
        foot_pos = self.body_pos_w[:, -self.foot_patch_num:, :]
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
        return roll.view(-1, self.foot_patch_num)
    
    
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
        return pitch.view(-1, self.foot_patch_num)
    
    @property
    def foot_yaw_angle(self)->torch.Tensor:
        """
        Returns:
            beta (torch.Tensor): Foot angle of attack in radian (num_env, num_contact_points)
        """
        foot_quat = self.foot_quat.reshape(-1, 4)
        yaw = torch.atan2(2*(foot_quat[:, 0]*foot_quat[:, 3] + foot_quat[:, 1]*foot_quat[:, 2]), 1-2*(foot_quat[:, 2]**2 + foot_quat[:, 3]**2))
        yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw)) #standardize to -pi to pi
        return yaw.view(-1, self.foot_patch_num)
    
    @property
    def foot_roll_yaw_matrix(self) -> torch.Tensor:
        """
        Returns:
            foot_roll_yaw_matrix (torch.Tensor): Rotation matrices computed from foot roll and yaw angles,
                                                with pitch fixed to zero.
                                                Shape: (N, foot_patch_num, 3, 3)
        """
        # Retrieve the batched roll and yaw angles.
        # (Make sure to use the correct attribute for roll; here we assume it's self.foot_roll_angle.)
        # Both roll and yaw should have shape (N, foot_patch_num)
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
        
        # Construct the rotation matrix for roll about the X-axis.
        # R_x = [ [1,      0,       0],
        #         [0,   cos(roll), -sin(roll)],
        #         [0,   sin(roll),  cos(roll)] ]
        R_x = torch.stack([
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, c_r, -s_r], dim=-1),
            torch.stack([zeros, s_r, c_r], dim=-1)
        ], dim=-2)  # R_x has shape (N, foot_patch_num, 3, 3)
        
        # Construct the rotation matrix for yaw about the Z-axis.
        # R_z = [ [cos(yaw), -sin(yaw), 0],
        #         [sin(yaw),  cos(yaw), 0],
        #         [   0,         0,     1] ]
        R_z = torch.stack([
            torch.stack([c_y, -s_y, zeros], dim=-1),
            torch.stack([s_y, c_y, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1)
        ], dim=-2)  # R_z has shape (N, foot_patch_num, 3, 3)
        
        # Compose the rotations.
        # For an intrinsic rotation order (roll followed by yaw) with pitch = 0,
        # the overall rotation matrix is given by:
        #     R = R_x * R_z
        # where each multiplication is performed for each (N, foot_patch_num) pair.
        foot_mat = torch.matmul(R_x, R_z)  # (N, foot_patch_num, 3, 3)
        
        return foot_mat
    
    @property
    def foot_beta_angle(self)->torch.Tensor:
        """
        Returns:
            beta (torch.Tensor): Foot angle of attack in radian (N, 12)
        """
        beta = -self.foot_pitch_angle
        return beta.view(-1, self.foot_patch_num)
    
    # gamma retrieved from velocity wrt local coordinate (where roll and yaw are aligned with body frame)
    @property
    def foot_gamma_angle(self)->torch.Tensor:
        """
        Commpute gamma using linear velocity wrt body frame (roll, yaw aligned with body frame) of each foot links
        Returns:
            gamma (torch.Tensor): Foot intrusion angle in radian (N, 12)
        """
        foot_vel = self.body_lin_vel_w[:, -self.foot_patch_num:, :].reshape(-1, 3)
        foot_vel = torch.bmm(self.foot_roll_yaw_matrix.reshape(-1, 3, 3).transpose(1,2), foot_vel[:, :, None]).squeeze(-1) # get velocity wrt local coordinate
        gamma = torch.atan2(-foot_vel[:, 2], foot_vel[:, 0])
        gamma = torch.atan2(torch.sin(gamma), torch.cos(gamma)) #standardize to -pi to pi
        
        # map gamma (-pi, pi) to (-pi/2, pi/2) due to RFT convention
        sign_mask = gamma > 0
        coordinate_mask = torch.abs(gamma) < torch.pi/2
        gamma[sign_mask*~coordinate_mask] = torch.pi - gamma[sign_mask*~coordinate_mask]
        gamma[~sign_mask*~coordinate_mask] = -torch.pi - gamma[~sign_mask*~coordinate_mask]
        
        return gamma.view(-1, self.foot_patch_num)
    
    # gamma retrieved from velocity wrt global coordiante
    # @property
    # def foot_gamma_angle(self)->torch.Tensor:
    #     """
    #     Commpute gamma using linear velocity of each foot links
    #     Returns:
    #         gamma (torch.Tensor): Foot intrusion angle in radian (N, 12)
    #     """
    #     foot_vel = self.body_lin_vel_w[:, -self.foot_patch_num:, :].reshape(-1, 3)
    #     gamma = torch.atan2(-foot_vel[:, 2], foot_vel[:, 0])
    #     gamma = torch.atan2(torch.sin(gamma), torch.cos(gamma)) #standardize to -pi to pi

    #     sign_mask = gamma > 0
    #     coordinate_mask = torch.abs(gamma) < torch.pi/2
    #     gamma[sign_mask*~coordinate_mask] = torch.pi - gamma[sign_mask*~coordinate_mask]
    #     gamma[~sign_mask*~coordinate_mask] = -torch.pi - gamma[~sign_mask*~coordinate_mask]
    #     return gamma.view(-1, self.foot_patch_num)
    
    @property
    def body_names(self)->list:
        return self.articulation.body_names
    
    
    # @property
    # def foot_gamma_angle_kinematics(self)->torch.Tensor:
    #     """
    #     Compute gamma using kinematics
    #     Returns:
    #         gamma (torch.Tensor): Foot intrusion angle in radian (N, 12)
    #     """
    #     foot_lin_vel = self.body_lin_vel_w[:, -14:-12, :]
    #     foot_ang_vel = self.body_ang_vel_w[:, -14:-12, :]
    #     foot_rel_pos_left = self.body_pos_w[:, -12:-6, :] - self.body_pos_w[:, -14:-13, :]
    #     foot_rel_pos_right = self.body_pos_w[:, -6:, :] - self.body_pos_w[:, -13:-12, :]
    #     foot_vel_left = foot_lin_vel[:, 0:1, :] + torch.cross(foot_ang_vel[:, 0:1, :], foot_rel_pos_left, dim=2)
    #     foot_vel_right = foot_lin_vel[:, 1:2, :] + torch.cross(foot_ang_vel[:, 1:2, :], foot_rel_pos_right, dim=2)
    #     foot_vel_mask_left = torch.norm(foot_vel_left, dim=2) > 0.1
    #     foot_vel_mask_right = torch.norm(foot_vel_right, dim=2) > 0.1
    #     gamma_left = torch.atan2(-foot_vel_left[:,:, 2], foot_vel_left[:, :, 0]) * foot_vel_mask_left
    #     gamma_right = torch.atan2(-foot_vel_right[:, :, 2], foot_vel_right[:, :, 0]) * foot_vel_mask_right
    #     return torch.cat((gamma_left, gamma_right), dim=1)