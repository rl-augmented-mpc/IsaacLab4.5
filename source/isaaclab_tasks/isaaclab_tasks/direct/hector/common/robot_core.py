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
            foot_depth (torch.Tensor): Foot penetration depth (N, 12)
        """
        foot_pos = self.body_pos_w[:, -self.foot_patch_num:, :]
        return foot_pos
    
    @property
    def foot_quat(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (N, 12)
        """
        foot_quat = self.body_quat_w[:, -self.foot_patch_num:, :]
        return foot_quat
    
    @property
    def foot_rot_mat(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (N, 12)
        """
        foot_quat = self.body_quat_w[:, -self.foot_patch_num:, :]
        foot_mat = matrix_from_quat(foot_quat.reshape(-1, 4))
        return foot_mat.reshape(-1, self.foot_patch_num, 3, 3)
        
    
    @property
    def foot_vel(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (N, 12)
        """
        foot_vel = self.body_lin_vel_w[:, -self.foot_patch_num:, :]
        return foot_vel
    
    @property
    def foot_accel(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (N, 12)
        """
        foot_vel = self.body_lin_acc_w[:, -self.foot_patch_num:, :]
        return foot_vel
    
    @property
    def foot_depth(self)->torch.Tensor:
        """
        Returns:
            foot_depth (torch.Tensor): Foot penetration depth (N, 12)
        """
        foot_pos = self.body_pos_w[:, -self.foot_patch_num:, :]
        return foot_pos[:, :, -1]
    
    @property
    def foot_yaw_angle(self)->torch.Tensor:
        """
        Returns:
            beta (torch.Tensor): Foot angle of attack in radian (N, 12)
        """
        foot_quat = self.body_quat_w[:, -self.foot_patch_num:, :].reshape(-1, 4)
        yaw = torch.atan2(2*(foot_quat[:, 2]*foot_quat[:, 0] + foot_quat[:, 1]*foot_quat[:, 3]), 1-2*(foot_quat[:, 0]**2 + foot_quat[:, 1]**2))
        yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw)) #standardize to -pi to pi
        return yaw.view(-1, self.foot_patch_num)
    
    @property
    def foot_beta_angle(self)->torch.Tensor:
        """
        Returns:
            beta (torch.Tensor): Foot angle of attack in radian (N, 12)
        """
        foot_quat = self.body_quat_w[:, -self.foot_patch_num:, :].reshape(-1, 4)
        pitch = torch.asin(2*(foot_quat[:, 0]*foot_quat[:, 2] - foot_quat[:, 3]*foot_quat[:, 1]))
        pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch)) #standardize to -pi to pi
        pitch *= -1
        return pitch.view(-1, self.foot_patch_num)
    
    @property
    def foot_gamma_angle(self)->torch.Tensor:
        """
        Commpute gamma using linear velocity of each foot links
        Returns:
            gamma (torch.Tensor): Foot intrusion angle in radian (N, 12)
        """
        foot_vel = self.body_lin_vel_w[:, -self.foot_patch_num:, :].reshape(-1, 3)
        gamma = torch.atan2(-foot_vel[:, 2], foot_vel[:, 0])
        gamma = torch.atan2(torch.sin(gamma), torch.cos(gamma)) #standardize to -pi to pi

        sign_mask = gamma > 0
        coordinate_mask = torch.abs(gamma) < torch.pi/2
        gamma[sign_mask*~coordinate_mask] = torch.pi - gamma[sign_mask*~coordinate_mask]
        gamma[~sign_mask*~coordinate_mask] = -torch.pi - gamma[~sign_mask*~coordinate_mask]
        return gamma.view(-1, self.foot_patch_num)
    
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