from typing import List, Tuple
import math
from inspect import isfunction
import torch
from dataclasses import dataclass
from isaaclab_tasks.direct.hector.common.curriculum import CurriculumRateSampler

@torch.jit.script
def compute_linear_penalty(
    value: torch.Tensor,
    scale: float = 1.0,
    min_value: float = torch.pi/36,
    max_value: float = torch.pi/2,
    ):
    value = (torch.abs(value) - min_value)
    value[value < 0] = 0
    value[value > max_value-min_value] = max_value-min_value
    value = torch.sum(value, -1)
    return scale*value

@torch.jit.script
def compute_gaussian_penalty(
    value: torch.Tensor,
    scale: float = 1.0,
    min_value: float = -1.0,
    max_value: float = 1.0,
    std: float = 0.5,
    ):
    value = torch.abs(value) - min_value
    value[value < 0] = 0
    value[value > 0] = (1 - torch.exp(-torch.square(value[value > 0])/(std**2))) * (max_value - min_value)
    value = torch.sum(value, -1)
    return scale*value

@dataclass
class OrientationRegularizationPenalty:
    roll_penalty_weight: float = 1.0
    pitch_penalty_weight: float = 1.0
    roll_range: tuple = (torch.pi/18, torch.pi/2)
    pitch_range: tuple = (torch.pi/18, torch.pi/2)
    
    def compute_penalty(self, root_quat: torch.Tensor)->tuple:
        roll = torch.atan2(2*(root_quat[:, 0]*root_quat[:, 1] + root_quat[:, 2]*root_quat[:, 3]), 1 - 2*(root_quat[:, 1]**2 + root_quat[:, 2]**2))
        roll = torch.atan2(torch.sin(roll), torch.cos(roll)) # standardize to -pi to pi
        
        pitch = torch.asin(2*(root_quat[:, 0]*root_quat[:, 2] - root_quat[:, 3]*root_quat[:, 1]))
        pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch)) # standardize to -pi to pi

        roll_penalty = self.roll_penalty_weight * roll.view(-1)
        pitch_penalty = self.pitch_penalty_weight * pitch.view(-1)
        
        # roll_penalty = self.roll_penalty_weight * compute_linear_penalty(roll.view(-1, 1), scale=1.0, min_value=self.roll_range[0], max_value=self.roll_range[1])
        # pitch_penalty = self.pitch_penalty_weight * compute_linear_penalty(pitch.view(-1, 1), scale=1.0, min_value=self.pitch_range[0], max_value=self.pitch_range[1])
        
        # roll_penalty = self.roll_penalty_weight * compute_gaussian_penalty(roll.view(-1, 1), scale=1.0, min_value=self.roll_range[0], max_value=self.roll_range[1], temperature=4.0)
        # pitch_penalty = self.pitch_penalty_weight * compute_gaussian_penalty(pitch.view(-1, 1), scale=1.0, min_value=self.pitch_range[0], max_value=self.pitch_range[1], temperature=4.0)
        
        return roll_penalty, pitch_penalty

@dataclass
class VelocityPenalty:
    velocity_penalty_weight: float = 1.0
    
    def compute_penalty(self, velocity: torch.Tensor)->torch.Tensor:
        penalty = torch.square(torch.norm(velocity[:, 2:], dim=1)) # vz 
        penalty = self.velocity_penalty_weight * penalty
        return penalty

@dataclass
class AngularVelocityPenalty:
    ang_velocity_penalty_weight: float = 1.0
    
    def compute_penalty(self, ang_velocity: torch.Tensor)->torch.Tensor:
        penalty = torch.square(torch.norm(ang_velocity[:, :2], dim=1)) # wx, wy
        penalty = self.ang_velocity_penalty_weight * penalty
        return penalty

@dataclass
class ActionRegularizationPenalty:
    action_penalty_weight: float = 1.0
    energy_penalty_weight: float = 1.0
    
    def compute_penalty(self,
                       action: torch.Tensor, 
                       previous_action: torch.Tensor)->tuple:
        action_penalty = torch.square(torch.norm(action - previous_action, dim=1))
        energy_penalty = torch.square(torch.norm(action, dim=1))
        
        action_penalty = self.action_penalty_weight*action_penalty
        energy_penalty = self.energy_penalty_weight*energy_penalty
        
        return action_penalty, energy_penalty

@dataclass
class CurriculumActionRegularizationPenalty:
    """
    Curriculum based action regularization penalty.
    """
    action_penalty_weight_start: float
    action_penalty_weight_end: float
    energy_penalty_weight_start: float
    energy_penalty_weight_end: float
    rate_sampler: CurriculumRateSampler
    
    def compute_penalty(self,
                       action: torch.Tensor, 
                       previous_action: torch.Tensor,
                       step:int = 0)->tuple:
        rate = self.rate_sampler.sample(step)
        action_penalty_weight = self.action_penalty_weight_start + rate * (self.action_penalty_weight_end - self.action_penalty_weight_start)
        energy_penalty_weight = self.energy_penalty_weight_start + rate * (self.energy_penalty_weight_end - self.energy_penalty_weight_start)

        action_penalty = torch.sum(torch.norm(action - previous_action, dim=1, keepdim=True), -1) # L2 norm of the difference
        energy_penalty = torch.sum(torch.norm(action, dim=1, keepdim=True), -1) # L2 norm of the action itself
        action_penalty = action_penalty_weight * action_penalty
        energy_penalty = energy_penalty_weight * energy_penalty
        return action_penalty, energy_penalty
    
@dataclass
class CurriculumTorqueRegularizationPenalty:
    """
    Curriculum based action regularization penalty.
    """
    torque_penalty_weight_start: float
    torque_penalty_weight_end: float
    rate_sampler: CurriculumRateSampler
    
    def compute_penalty(self,
                       action: torch.Tensor,
                       step:int = 0)->torch.Tensor:
        rate = self.rate_sampler.sample(step)
        energy_penalty_weight = self.torque_penalty_weight_start + rate * (self.torque_penalty_weight_end - self.torque_penalty_weight_start)

        energy_penalty = torch.sum(torch.norm(action, dim=1, keepdim=True), -1) # L2 norm
        energy_penalty = energy_penalty_weight * energy_penalty
        return energy_penalty

@dataclass
class FootDistanceRegularizationPenalty:
    foot_distance_penalty_weight: float = 1.0
    foot_distance_bound: tuple = (0.0, 0.5)
    
    def compute_penalty(self, left_foot_pos_b:torch.Tensor, right_foot_pos_b:torch.Tensor)->torch.Tensor:
        foot_distance = left_foot_pos_b[:, 1:2] - right_foot_pos_b[:, 1:2]
        # foot_distance_penalty = compute_linear_penalty(foot_distance.view(-1, 1), scale=1.0, min_value=self.foot_distance_bound[0], max_value=self.foot_distance_bound[1])
        # foot_distance_penalty = compute_gaussian_penalty(foot_distance.view(-1, 1), scale=1.0, min_value=self.foot_distance_bound[0], max_value=self.foot_distance_bound[1], temperature=4.0)
        foot_distance_penalty = self.foot_distance_penalty_weight*foot_distance.view(-1)
        return foot_distance_penalty

@dataclass
class FootAnglePenalty:
    foot_angle_penalty_weight: float = 1.0
    def compute_penalty(self, foot_angle:torch.Tensor)->torch.Tensor:
        foot_angle_penalty = self.foot_angle_penalty_weight * foot_angle.sum(dim=1)
        return foot_angle_penalty
    

@dataclass
class TerminationPenalty:
    termination_weight: float = 1.0
    def compute_penalty(self, reset_terminated:torch.Tensor)->torch.Tensor:
        return self.termination_weight * reset_terminated

@dataclass
class JointPenalty:
    joint_penalty_weight: float = 1.0
    joint_pos_bound: tuple = (torch.pi/3, torch.pi/2)
    
    def compute_penalty(self, joint_pos: torch.Tensor)->torch.Tensor:
        joint_pos = torch.abs(joint_pos).view(-1)
        joint_penalty = self.joint_penalty_weight * joint_pos #L1

        # joint_penalty = compute_linear_penalty(joint_pos.view(-1, 1), scale=1.0, min_value=self.joint_pos_bound[0], max_value=self.joint_pos_bound[1])
        # joint_penalty = compute_gaussian_penalty(joint_pos, scale=1.0, min_value=self.joint_pos_bound[0], max_value=self.joint_pos_bound[1], temperature=2.0)
        # joint_penalty = self.joint_penalty_weight*joint_penalty

        return joint_penalty

@dataclass
class FeetSlidePenalty:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. 
    This ensures that the agent is penalized only when the feet are in contact with the ground.
    """
    feet_slide_weight: float = 1.0
    
    def compute_penalty(self, foot_velocity: torch.Tensor, contact: torch.Tensor)->torch.Tensor:
        feet_slide_penalty = torch.sum(torch.norm(foot_velocity, 2) * contact, dim=1) # L1
        # feet_slide_penalty = torch.sum(torch.square(torch.norm(foot_velocity, 2) * contact), dim=1) # L2
        # feet_slide_penalty = compute_gaussian_penalty(feet_slide_penalty.view(-1, 1), scale=1.0, min_value=2.0, max_value=4.0, temperature=4.0)
        # feet_slide_penalty = compute_linear_penalty(feet_slide_penalty.view(-1, 1), scale=1.0, min_value=2.0, max_value=4.0)
        feet_slide_penalty = self.feet_slide_weight*feet_slide_penalty
        return feet_slide_penalty

@dataclass
class ContactLocationPenalty:
    contact_location__penalty_weight: float = 1.0
    
    def compute_penalty(self, ground_gradient: torch.Tensor)->torch.Tensor:
        penalty = self.contact_location__penalty_weight * torch.abs(ground_gradient.view(-1))
        return penalty

@dataclass
class ActionSaturationPenalty:
    action_penalty_weight: float = 1.0
    action_bound: tuple = (0.9, 1.0)
    
    def compute_penalty(self, action: torch.Tensor)->torch.Tensor:
        action_penalty = compute_linear_penalty(action, scale=1.0, min_value=self.action_bound[0], max_value=self.action_bound[1])
        # action_penalty = compute_gaussian_penalty(action, scale=1.0, min_value=self.action_bound[0], max_value=self.action_bound[1], temperature=2.0)
        action_penalty = self.action_penalty_weight*action_penalty
        return action_penalty


### NOT USED ####

@dataclass
class TwistPenalty:
    vx_bound: tuple = (0.0, 1.0)
    vy_bound: tuple = (0.0, 1.0)
    wz_bound: tuple = (0.0, 1.0)
    vx_penalty_weight: float = 1.0
    vy_penalty_weight: float = 1.0
    wz_penalty_weight: float = 1.0
    
    def compute_penalty(self, root_lin_vel_b: torch.Tensor, root_ang_vel_b: torch.Tensor)->tuple:
        # vx_penalty = compute_linear_penalty(root_lin_vel_b[:, :1].view(-1, 1), scale=1.0, min_value=self.vx_bound[0], max_value=self.vx_bound[1])
        # vy_penalty = compute_linear_penalty(root_lin_vel_b[:, 1:2].view(-1, 1), scale=1.0, min_value=self.vy_bound[0], max_value=self.vy_bound[1])
        # wz_penalty = compute_linear_penalty(root_ang_vel_b[:, -1:].view(-1, 1), scale=1.0, min_value=self.wz_bound[0], max_value=self.wz_bound[1])
        vx_penalty = compute_gaussian_penalty(torch.abs(root_lin_vel_b[:, :1]), scale=1.0, min_value=self.vx_bound[0], max_value=self.vx_bound[1], temperature=4.0)
        vy_penalty = compute_gaussian_penalty(torch.abs(root_lin_vel_b[:, 1:2]), scale=1.0, min_value=self.vy_bound[0], max_value=self.vy_bound[1], temperature=4.0)
        wz_penalty = compute_gaussian_penalty(torch.abs(root_ang_vel_b[:, -1:]), scale=1.0, min_value=self.wz_bound[0], max_value=self.wz_bound[1], temperature=4.0)
        
        vx_penalty = self.vx_penalty_weight*vx_penalty
        vy_penalty = self.vy_penalty_weight*vy_penalty
        wz_penalty = self.wz_penalty_weight*wz_penalty
        
        return vx_penalty, vy_penalty, wz_penalty

@dataclass
class VelocityTrackingPenalty:
    roll_deviation_weight: float = 1.0
    pitch_deviation_weight: float = 1.0
    action_penalty_weight: float = 1.0
    energy_penalty_weight: float = 1.0
    foot_energy_penalty_weight: float = 1.0
    roll_range: tuple = (torch.pi/18, torch.pi/2)
    pitch_range: tuple = (torch.pi/18, torch.pi/2)
    
    def compute_penalty(self, 
                       root_quat: torch.Tensor, 
                       action: torch.Tensor, 
                       previous_action: torch.Tensor)->tuple:
        
        roll = torch.atan2(2*(root_quat[:, 0]*root_quat[:, 1] + root_quat[:, 2]*root_quat[:, 3]), 1 - 2*(root_quat[:, 1]**2 + root_quat[:, 2]**2))
        roll = torch.atan2(torch.sin(roll), torch.cos(roll)) # standardize to -pi to pi
        
        pitch = torch.asin(2*(root_quat[:, 0]*root_quat[:, 2] - root_quat[:, 3]*root_quat[:, 1]))
        pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch)) # standardize to -pi to pi
        
        # roll_penalty = compute_linear_penalty(roll.view(-1, 1), scale=1.0, min_value=self.roll_range[0], max_value=self.roll_range[1])
        # pitch_penalty = compute_linear_penalty(pitch.view(-1, 1), scale=1.0, min_value=self.pitch_range[0], max_value=self.pitch_range[1])
        roll_penalty = compute_gaussian_penalty(roll.view(-1, 1), scale=1.0, min_value=self.roll_range[0], max_value=self.roll_range[1])
        pitch_penalty = compute_gaussian_penalty(pitch.view(-1, 1), scale=1.0, min_value=self.pitch_range[0], max_value=self.pitch_range[1])
        action_penalty = torch.sum(torch.abs(action - previous_action), -1)
        energy_penalty = torch.sum(torch.abs(action), -1)
        foot_height_energy_penalty = torch.sum(torch.abs(action[:, -1:]), -1)
        
        roll_penalty = self.roll_deviation_weight*roll_penalty
        pitch_penalty = self.pitch_deviation_weight*pitch_penalty
        action_penalty = self.action_penalty_weight*action_penalty
        energy_penalty = self.energy_penalty_weight*energy_penalty
        foot_height_energy_penalty = self.foot_energy_penalty_weight*foot_height_energy_penalty
        
        return roll_penalty, pitch_penalty, action_penalty, energy_penalty, foot_height_energy_penalty
    
@dataclass
class GoalTrackingPenalty:
    roll_deviation_weight: float = 1.0
    pitch_deviation_weight: float = 1.0
    yaw_deviation_weight: float = 1.0
    ankle_pitch_deviation_weight: float = 1.0
    action_penalty_weight: float = 1.0
    energy_penalty_weight: float = 1.0
    terminated_weight: float = 1.0
    
    
    def compute_penalty(self, 
                       root_quat: torch.Tensor, 
                       action: torch.Tensor, 
                       previous_action: torch.Tensor, 
                       ankle_pitch: torch.Tensor,
                       reset_terminated: torch.Tensor)->tuple:
        
        roll = torch.atan2(2*(root_quat[:, 0]*root_quat[:, 1] + root_quat[:, 2]*root_quat[:, 3]), 1 - 2*(root_quat[:, 1]**2 + root_quat[:, 2]**2))
        roll = torch.atan2(torch.sin(roll), torch.cos(roll)) # standardize to -pi to pi
        
        pitch = torch.asin(2*(root_quat[:, 0]*root_quat[:, 2] - root_quat[:, 3]*root_quat[:, 1]))
        pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch)) # standardize to -pi to pi
        
        yaw = torch.atan2(2*(root_quat[:, 0]*root_quat[:, 3] + root_quat[:, 1]*root_quat[:, 2]), 1 - 2*(root_quat[:, 2]**2 + root_quat[:, 3]**2)).view(-1, 1)
        yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw)) # standardize to -pi to pi
        
        # roll_penalty = compute_linear_penalty(roll.view(-1, 1), scale=1.0, min_value=torch.pi/18, max_value=torch.pi/2)
        # pitch_penalty = compute_linear_penalty(pitch.view(-1, 1), scale=1.0, min_value=torch.pi/18, max_value=torch.pi/2)
        # yaw_penalty = compute_linear_penalty(yaw.view(-1, 1), scale=1.0, min_value=torch.pi/18, max_value=torch.pi)
        roll_penalty = compute_gaussian_penalty(roll.view(-1, 1), scale=1.0, min_value=torch.pi/18, max_value=torch.pi/2)
        pitch_penalty = compute_gaussian_penalty(pitch.view(-1, 1), scale=1.0, min_value=torch.pi/18, max_value=torch.pi/2)
        yaw_penalty = compute_gaussian_penalty(yaw.view(-1, 1), scale=1.0, min_value=torch.pi/18, max_value=torch.pi)
        
        action_penalty = torch.sum(torch.abs(action - previous_action), -1)
        energy_penalty = torch.sum(torch.abs(action), -1)
        
        # ankle_pitch_penalty = compute_linear_penalty(ankle_pitch.view(-1, 2), scale=1.0, min_value=torch.pi/9, max_value=torch.pi/2)
        ankle_pitch_penalty = compute_gaussian_penalty(ankle_pitch.view(-1, 2), scale=1.0, min_value=torch.pi/9, max_value=torch.pi/2)
        terminated_penalty = reset_terminated.float().view(-1)
        
        roll_penalty = self.roll_deviation_weight*roll_penalty
        pitch_penalty = self.pitch_deviation_weight*pitch_penalty
        yaw_penalty = self.yaw_deviation_weight*yaw_penalty
        action_penalty = self.action_penalty_weight*action_penalty
        energy_penalty = self.energy_penalty_weight*energy_penalty
        ankle_pitch_penalty = self.ankle_pitch_deviation_weight*ankle_pitch_penalty
        terminated_penalty = self.terminated_weight*terminated_penalty
        
        return roll_penalty, pitch_penalty, yaw_penalty, action_penalty, energy_penalty, ankle_pitch_penalty, terminated_penalty

@dataclass
class PositionConePenalty:
    position_deviation_weight: float = 1.0
    anchor_position: tuple = (0.0, 0.0) # anchor position offset from target origin
    
    def compute_penalty(self,
                        root_pos: torch.Tensor, 
                        reference_pos: torch.Tensor)->torch.Tensor:
        
        relative_angle = torch.atan2(root_pos[:, 1]-reference_pos[:, 1], -(root_pos[:, 0]-reference_pos[:, 0]))
        relative_angle = torch.atan2(torch.sin(relative_angle), torch.cos(relative_angle)) # -pi to pi
        angle_penalty = compute_linear_penalty(relative_angle.view(-1, 1), scale=1.0, min_value=torch.pi/36, max_value=torch.pi/3)
        angle_penalty = self.position_deviation_weight*angle_penalty
        return angle_penalty

### NOT USED ####


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # position_x = torch.arange(0.0, 3.0, 0.1)
    # position_y = torch.arange(-0.5, 0.5, 0.1)
    # X, Y = torch.meshgrid(position_x, position_y, indexing="ij")
    
    # goal_position = torch.tensor([3.0, 0.0]).view(1, -1)
    # goal_yaw = torch.tensor([0.0]).view(1, -1)
    
    # position = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)
    # root_height = 0.55 * torch.ones(position.shape[0], 1)
    # root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(position.shape[0], 1)
    # yaw = 3.14/6 * torch.ones(position.shape[0], 1)
    # action = torch.zeros(position.shape[0], 12)
    # previous_action = torch.zeros(position.shape[0], 12)
    # ankle_pitch = torch.zeros(position.shape[0], 2)
    # reset_terminated = torch.zeros(position.shape[0], 1)
    
    
    # goal_tracking_penalty = GoalTrackingPenalty(
    #     roll_deviation_weight = 1.0,
    #     pitch_deviation_weight= 1.0,
    #     yaw_deviation_weight= 1.0,
    #     ankle_pitch_deviation_weight = 1.0,
    #     action_penalty_weight = 1.0,
    #     energy_penalty_weight = 1.0,
    #     terminated_weight = 1.0,
    # )
    
    # position_cone_penalty = PositionConePenalty(
    #     position_deviation_weight = 1.0,
    #     anchor_position=(0.1, 0.0),
    # )
    
    # roll_penalty, pitch_penalty, yaw_penalty, action_penalty, energy_penalty, ankle_pitch_penalty, terminated_penalty \
    #     = goal_tracking_penalty.compute_penalty(root_quat, action, previous_action, ankle_pitch, reset_terminated)
        
    # cone_penalty = position_cone_penalty.compute_penalty(position, goal_position)
    
    # cone_penalty_grid = cone_penalty.reshape(X.shape)
    
    # plt.figure()
    # plt.imshow(cone_penalty_grid, cmap="jet", origin="lower")
    # plt.colorbar()
    # plt.xlabel("Y")
    # plt.ylabel("X")
    # plt.show()
    
    min_value = 0.5
    max_value = 1.0
    value = torch.linspace(-2.0, 2.0, 200).view(-1, 1)
    penalty = compute_linear_penalty(value, scale=1.0, min_value=min_value, max_value=max_value)
    plt.plot(value, penalty, label="linear")
    penalty = compute_gaussian_penalty(value, scale=1.0, min_value=min_value, max_value=max_value, temperature=2.0)
    plt.plot(value, penalty, label="gaussian")
    plt.legend()
    plt.show()