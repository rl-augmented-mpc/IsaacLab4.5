
import torch
from dataclasses import dataclass

@torch.jit.script
def compute_linear_reward(
    error: torch.Tensor,
    scale: float = 1.0,
    )->torch.Tensor:
    return -scale*error

@torch.jit.script
def compute_square_reward(
    error: torch.Tensor,
    scale: float = 1.0,
    )->torch.Tensor:
    return scale/(1.0+error)

@torch.jit.script
def compute_exponential_reward(
    error: torch.Tensor, # normed error
    temperature: float = 1.0,
    scale: float = 1.0,
    )->torch.Tensor:
    return scale*torch.exp(-temperature*error)

# gaussian kernel with mean=0
@torch.jit.script
def compute_gaussian_reward(
    error: torch.Tensor, # normed error
    temperature: float = 1.0,
    scale: float = 1.0,
    )->torch.Tensor:
    return scale*torch.exp(-temperature*torch.square(error))

@dataclass
class AliveReward:
    alive_weight: float = 1.0
    
    def compute_reward(self, reset_terminated:torch.Tensor, current_step:torch.Tensor, max_step:int)->torch.Tensor:
        return self.alive_weight * ((1 - reset_terminated.float()) * (current_step/max_step))

@dataclass
class SagittalFPSimilarityReward:
    fp_similarity_weight: float = 1.0
    fp_similarity_coeff: float = 0.25
    fp_reward_mode:str="exponential"
    
    def compute_reward(self, residual_sagittal_fp:torch.Tensor):
        fp_error = 100 * torch.norm(residual_sagittal_fp[:, 0:1] - residual_sagittal_fp[:, 1:], dim=1).view(-1, 1) # convert to m to cm for tighter reward
        if self.fp_reward_mode == "linear":
            fp_reward = compute_linear_reward(fp_error, scale=1.0)
        elif self.fp_reward_mode == "square":
            fp_reward = compute_square_reward(fp_error, scale=1.0)
        elif self.fp_reward_mode == "exponential":
            fp_reward = compute_exponential_reward(fp_error, scale=1.0, temperature=self.fp_similarity_coeff)
        elif self.fp_reward_mode == "gaussian":
            fp_reward = compute_gaussian_reward(fp_error, scale=1.0, temperature=self.fp_similarity_coeff)
        
        fp_reward = self.fp_similarity_weight * fp_reward.squeeze(1)
        return fp_reward

@dataclass
class VelocityTrackingReward:
    height_similarity_weight: float = 1.0
    lin_vel_similarity_weight: float = 1.0
    ang_vel_similarity_weight: float = 1.0
    height_similarity_coeff: float = 0.25
    lin_vel_similarity_coeff: float = 0.25
    ang_vel_similarity_coeff: float = 0.25
    height_reward_mode:str="square"
    lin_vel_reward_mode:str="square"
    ang_vel_reward_mode:str="square"
    
    def __post_init__(self):
        assert self.height_reward_mode in ["linear", "square", "exponential", "gaussian"]
        assert self.lin_vel_reward_mode in ["linear", "square", "exponential", "gaussian"]
        assert self.ang_vel_reward_mode in ["linear", "square", "exponential", "gaussian"]
    
    def compute_reward(self, 
                       root_pos: torch.Tensor, 
                       root_lin_vel_b: torch.Tensor, 
                       root_ang_vel_b: torch.Tensor,
                       reference_height:torch.Tensor,
                       desired_root_lin_vel_b: torch.Tensor, 
                       desired_root_ang_vel_b: torch.Tensor)->tuple:
        """
        root_pos: (num_envs, 3)
        root_lin_vel_b: (num_envs, 3)
        root_ang_vel_b: (num_envs, 3)
        desired_root_lin_vel_b: (num_envs, 2)
        desired_root_ang_vel_b: (num_envs, 1)
        """
        
        height_error = torch.abs(reference_height.view(-1, 1) - root_pos[:, 2:3]).view(-1, 1)
        lin_vel_error = torch.norm(desired_root_lin_vel_b - root_lin_vel_b[:, :2], dim=1).view(-1, 1) # norm(error_vx, error_vy)
        ang_vel_error = torch.abs(desired_root_ang_vel_b - root_ang_vel_b[:, 2:]).view(-1, 1)
        
        if self.height_reward_mode == "linear":
            height_reward = torch.sum(compute_linear_reward(height_error, scale=1.0), -1)
        elif self.height_reward_mode == "square":
            height_reward = torch.sum(compute_square_reward(height_error, scale=1.0), -1)
        elif self.height_reward_mode == "exponential":
            height_reward = torch.sum(compute_exponential_reward(height_error, scale=1.0, temperature=self.height_similarity_coeff), -1)
        elif self.height_reward_mode == "gaussian":
            height_reward = torch.sum(compute_gaussian_reward(height_error, scale=1.0, temperature=self.height_similarity_coeff), -1)
        
        if self.lin_vel_reward_mode == "linear":
            lin_vel_reward = torch.sum(compute_linear_reward(lin_vel_error, scale=1.0), -1)
        elif self.lin_vel_reward_mode == "square":
            lin_vel_reward = torch.sum(compute_square_reward(lin_vel_error, scale=1.0), -1)
        elif self.lin_vel_reward_mode == "exponential":
            lin_vel_reward = torch.sum(compute_exponential_reward(lin_vel_error, scale=1.0, temperature=self.lin_vel_similarity_coeff), -1)
        elif self.lin_vel_reward_mode == "gaussian":
            lin_vel_reward = torch.sum(compute_gaussian_reward(lin_vel_error, scale=1.0, temperature=self.lin_vel_similarity_coeff), -1)
        
        if self.ang_vel_reward_mode == "linear":
            ang_vel_reward = torch.sum(compute_linear_reward(torch.square(ang_vel_error), scale=1.0), -1)
        elif self.ang_vel_reward_mode == "square":
            ang_vel_reward = torch.sum(compute_square_reward(torch.square(ang_vel_error), scale=1.0), -1)
        elif self.ang_vel_reward_mode == "exponential":
            ang_vel_reward = torch.sum(compute_exponential_reward(ang_vel_error, scale=1.0, temperature=self.ang_vel_similarity_coeff), -1)
        elif self.ang_vel_reward_mode == "gaussian":
            ang_vel_reward = torch.sum(compute_gaussian_reward(ang_vel_error, scale=1.0, temperature=self.ang_vel_similarity_coeff), -1)
        
        height_reward = self.height_similarity_weight*height_reward
        lin_vel_reward = self.lin_vel_similarity_weight*lin_vel_reward
        ang_vel_reward = self.ang_vel_similarity_weight*ang_vel_reward
        
        return height_reward, lin_vel_reward, ang_vel_reward


@dataclass
class PoseTrackingReward:
    position_weight: float = 1.0
    yaw_weight: float = 1.0
    position_coeff: float = 0.25
    yaw_coeff: float = 0.25
    position_reward_mode:str="square"
    yaw_reward_mode:str="square"
    
    def __post_init__(self):
        assert self.position_reward_mode in ["linear", "square", "exponential", "gaussian"]
        assert self.yaw_reward_mode in ["linear", "square", "exponential", "gaussian"]
    
    def compute_reward(self, 
                       root_pos: torch.Tensor,
                       root_yaw: torch.Tensor, 
                       ref_root_pos: torch.Tensor, 
                       ref_yaw: torch.Tensor)->tuple:
        
        position_error = torch.norm(ref_root_pos - root_pos, dim=1).view(-1, 1)
        yaw_error = torch.abs(ref_yaw.view(-1) - root_yaw.view(-1)).view(-1, 1)
        
        if self.position_reward_mode == "linear":
            position_reward = torch.sum(compute_linear_reward(position_error, scale=1.0), -1)
        elif self.position_reward_mode == "square":
            position_reward = torch.sum(compute_square_reward(position_error, scale=1.0), -1)
        elif self.position_reward_mode == "exponential":
            position_reward = torch.sum(compute_exponential_reward(position_error, scale=1.0, temperature=self.position_coeff), -1)
        elif self.position_reward_mode == "gaussian":
            position_reward = torch.sum(compute_gaussian_reward(position_error, scale=1.0, temperature=self.position_coeff), -1)
        
        if self.yaw_reward_mode == "linear":
            yaw_reward = torch.sum(compute_linear_reward(yaw_error, scale=1.0), -1)
        elif self.yaw_reward_mode == "square":
            yaw_reward = torch.sum(compute_square_reward(yaw_error, scale=1.0), -1)
        elif self.yaw_reward_mode == "exponential":
            yaw_reward = torch.sum(compute_exponential_reward(yaw_error, scale=1.0, temperature=self.yaw_coeff), -1)
        elif self.yaw_reward_mode == "gaussian":
            yaw_reward = torch.sum(compute_gaussian_reward(yaw_error, scale=1.0, temperature=self.yaw_coeff), -1)
        
        position_reward = self.position_weight*position_reward
        yaw_reward = self.yaw_weight*yaw_reward
        
        return position_reward, yaw_reward

@dataclass
class SwingFootTrackingReward:
    swing_foot_weight: float = 1.0
    swing_foot_coeff: float = 0.25
    swing_foot_reward_mode:str="square"
    
    def __post_init__(self):
        assert self.swing_foot_reward_mode in ["linear", "square", "exponential", "gaussian"]
    
    def compute_reward(self, 
                       swing_foot_pos: torch.Tensor, 
                       ref_swing_foot_pos: torch.Tensor)->torch.Tensor:
        
        swing_foot_error = torch.norm(ref_swing_foot_pos - swing_foot_pos, dim=1, keepdim=True)
        if self.swing_foot_reward_mode == "linear":
            swing_foot_reward = torch.sum(compute_linear_reward(swing_foot_error, scale=1.0), 1)
        elif self.swing_foot_reward_mode == "square":
            swing_foot_reward = torch.sum(compute_square_reward(swing_foot_error, scale=1.0), 1)
        elif self.swing_foot_reward_mode == "exponential":
            swing_foot_reward = torch.sum(compute_exponential_reward(swing_foot_error, scale=1.0, temperature=self.swing_foot_coeff), 1)
        elif self.swing_foot_reward_mode == "gaussian":
            swing_foot_reward = torch.sum(compute_gaussian_reward(swing_foot_error, scale=1.0, temperature=self.swing_foot_coeff), 1)
        swing_foot_reward = self.swing_foot_weight * swing_foot_reward
        return swing_foot_reward # type: ignore

@dataclass
class GoalTrackingReward:
    height_similarity_weight: float = 1.0
    goal_pos_dist_weight: float = 1.0
    goal_yaw_dist_weight: float = 1.0
    height_reward_mode:str="square"
    goal_pos_dist_reward_mode:str="square"
    goal_yaw_dist_reward_mode:str="square"
    
    def __post_init__(self):
        assert self.height_reward_mode in ["linear", "square", "exponential", "gaussian"]
        assert self.goal_pos_dist_reward_mode in ["linear", "square", "exponential", "gaussian"]
        assert self.goal_yaw_dist_reward_mode in ["linear", "square", "exponential", "gaussian"]
    
    def compute_reward(self, 
                       root_height: torch.Tensor, #(num_envs, 1)
                       goal_pos_dist: torch.Tensor, #(num_envs, 2)
                       goal_yaw_dist: torch.Tensor, #(num_envs, 1)
                       reference_height:float)->tuple: 
        
        height_error = torch.abs(reference_height - root_height).view(-1, 1)
        goal_pos_error = torch.norm(goal_pos_dist, dim=1).view(-1, 1)
        goal_yaw_error = torch.abs(goal_yaw_dist).view(-1, 1)
        
        if self.height_reward_mode == "linear":
            height_reward = torch.sum(compute_linear_reward(height_error, scale=1.0), -1)
        elif self.height_reward_mode == "square":
            height_reward = torch.sum(compute_square_reward(height_error, scale=1.0), -1)
        elif self.height_reward_mode == "exponential":
            height_reward = torch.sum(compute_exponential_reward(height_error, scale=1.0, temperature=0.5), -1)
        elif self.height_reward_mode == "gaussian":
            height_reward = torch.sum(compute_gaussian_reward(height_error, scale=1.0, temperature=0.5), -1)
        
        if self.goal_pos_dist_reward_mode == "linear":
            goal_pos_dist_reward = torch.sum(compute_linear_reward(goal_pos_error, scale=1.0), -1)
        elif self.goal_pos_dist_reward_mode == "square":
            goal_pos_dist_reward = torch.sum(compute_square_reward(goal_pos_error, scale=1.0), -1)
        elif self.goal_pos_dist_reward_mode == "exponential":
            goal_pos_dist_reward = torch.sum(compute_exponential_reward(goal_pos_error, scale=1.0, temperature=0.05), -1)
        elif self.goal_pos_dist_reward_mode == "gaussian":
            goal_pos_dist_reward = torch.sum(compute_gaussian_reward(goal_pos_error, scale=1.0, temperature=0.05), -1)
        
        if self.goal_yaw_dist_reward_mode == "linear":
            goal_yaw_dist_reward = torch.sum(compute_linear_reward(goal_yaw_error, scale=1.0), -1)
        elif self.goal_yaw_dist_reward_mode == "square":
            goal_yaw_dist_reward = torch.sum(compute_square_reward(goal_yaw_error, scale=1.0), -1)
        elif self.goal_yaw_dist_reward_mode == "exponential":
            goal_yaw_dist_reward = torch.sum(compute_exponential_reward(goal_yaw_error, scale=1.0, temperature=0.05), -1)
        elif self.goal_yaw_dist_reward_mode == "gaussian":
            goal_yaw_dist_reward = torch.sum(compute_gaussian_reward(goal_yaw_error, scale=1.0, temperature=0.05), -1)
        
        height_reward = self.height_similarity_weight*height_reward
        goal_pos_dist_reward = self.goal_pos_dist_weight*goal_pos_dist_reward
        goal_yaw_dist_reward = self.goal_yaw_dist_weight*goal_yaw_dist_reward
        
        return height_reward, goal_pos_dist_reward, goal_yaw_dist_reward

@dataclass
class ContactTrackingReward:
    contact_similarity_weight: float = 1.0
    contact_similarity_coeff: float = 0.25
    contact_reward_mode:str="square"
    
    def __post_init__(self):
        assert self.contact_reward_mode in ["linear", "square", "exponential", "gaussian"]
    
    def compute_reward(self, 
                       gait_contact: torch.Tensor, #(num_envs, 2)
                       gt_contact: torch.Tensor)->torch.Tensor: 
        
        contact_error = torch.norm(gait_contact - gt_contact, dim=1)
        if self.contact_reward_mode == "linear":
            contact_reward = compute_linear_reward(contact_error, scale=1.0)
        elif self.contact_reward_mode == "square":
            contact_reward = compute_square_reward(contact_error, scale=1.0)
        elif self.contact_reward_mode == "exponential":
            contact_reward = compute_exponential_reward(contact_error, scale=1.0, temperature=self.contact_similarity_coeff)
        elif self.contact_reward_mode == "gaussian":
            contact_reward = compute_gaussian_reward(contact_error, scale=1.0, temperature=self.contact_similarity_coeff)
        
        weight = self.contact_similarity_weight * torch.ones_like(contact_error)
        contact_reward = weight * contact_reward
        
        return contact_reward


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # position_x = torch.arange(0.0, 3.0, 0.1)
    # position_y = torch.arange(-0.5, 0.5, 0.1)
    # X, Y = torch.meshgrid(position_x, position_y, indexing="ij")
    
    # goal_position = torch.tensor([3.0, 0.0]).view(1, -1)
    # goal_yaw = torch.tensor([0.0]).view(1, -1)
    
    # position = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)
    # root_height = 0.55 * torch.ones(position.shape[0], 1)
    # yaw = 3.14/6 * torch.ones(position.shape[0], 1)
    # ankel_pitch = torch.zeros(position.shape[0], 2)
    # reference_height = 0.55
    
    # goal_pos_dist = goal_position - position
    # goal_yaw_dist = goal_yaw - yaw
    
    # goal_tracking_reward = GoalTrackingReward(
    #     height_similarity_weight= 0.0,
    #     goal_pos_dist_weight= 1.00,
    #     goal_yaw_dist_weight= 0.0,
    #     height_reward_mode="square",
    #     goal_pos_dist_reward_mode="square",
    #     goal_yaw_dist_reward_mode="exponential"
    # )
    
    # height_reward, goal_pos_dist_reward, goal_yaw_dist_reward = \
    #     goal_tracking_reward.compute_reward(root_height, goal_pos_dist, goal_yaw_dist, reference_height)
    
    
    # height_reward_grid = height_reward.reshape(X.shape)
    # goal_pos_dist_reward_grid = goal_pos_dist_reward.reshape(X.shape)
    # goal_yaw_dist_reward_grid = goal_yaw_dist_reward.reshape(X.shape)
    
    # total_reward_grid = height_reward_grid + goal_pos_dist_reward_grid + goal_yaw_dist_reward_grid
    
    # plt.figure()
    # plt.imshow(total_reward_grid, cmap="jet", origin="lower")
    # plt.colorbar()
    # plt.xlabel("Y")
    # plt.ylabel("X")
    # plt.show()
    
    
    
    # check error reward 
    error = torch.linspace(0.0, 0.05, 100) * 100
    gaussian_reward = compute_gaussian_reward(error, temperature=2.0, scale=1.0)
    exp_reward = compute_exponential_reward(error, temperature=2.0, scale=1.0)
    plt.plot(error, exp_reward, label="Exponential")
    plt.plot(error, gaussian_reward, label="Gaussian")
    plt.xlabel("Error")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()