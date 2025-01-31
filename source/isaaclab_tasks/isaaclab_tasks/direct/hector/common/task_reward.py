import torch
from dataclasses import dataclass

@torch.jit.script
def compute_linear_reward(
    error: torch.Tensor,
    scale: float = 1.0,
    ):
    return -scale*error

@torch.jit.script
def compute_square_reward(
    error: torch.Tensor,
    scale: float = 1.0,
    ):
    return scale/(1.0+error)

# gaussian kernel with mean=0
@torch.jit.script
def compute_exponential_reward(
    error: torch.Tensor, # normed error
    temperature: float = 1.0,
    scale: float = 1.0,
    ):
    return scale*torch.exp(-error/temperature)

@dataclass
class AliveReward:
    alive_weight: float = 1.0
    
    def compute_reward(self, reset_terminated:torch.Tensor, current_step:torch.Tensor, max_step:int)->torch.Tensor:
        return self.alive_weight * ((1 - reset_terminated.float()) * (current_step/max_step))

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
        assert self.height_reward_mode in ["linear", "square", "exponential"]
        assert self.lin_vel_reward_mode in ["linear", "square", "exponential"]
        assert self.ang_vel_reward_mode in ["linear", "square", "exponential"]
    
    def compute_reward(self, 
                       root_pos: torch.Tensor, 
                       root_lin_vel_b: torch.Tensor, 
                       root_ang_vel_b: torch.Tensor,
                       reference_height:float,
                       desired_root_lin_vel_b: torch.Tensor, 
                       desired_root_ang_vel_b: torch.Tensor)->tuple:
        
        # height_error = reference_height*torch.ones(root_pos.shape[0], 1, device=root_pos.device) - root_pos[:, 2:3]
        height_error = torch.abs(reference_height - root_pos[:, 2:3])
        lin_vel_error = torch.norm(desired_root_lin_vel_b - root_lin_vel_b[:, :2], dim=1).view(-1, 1)
        ang_vel_error = torch.norm(desired_root_ang_vel_b - root_ang_vel_b[:, 2:], dim=1).view(-1, 1)
        
        if self.height_reward_mode == "linear":
            height_reward = torch.sum(compute_linear_reward(height_error, scale=1.0), -1)
        elif self.height_reward_mode == "square":
            height_reward = torch.sum(compute_square_reward(height_error, scale=1.0), -1)
        elif self.height_reward_mode == "exponential":
            height_reward = torch.sum(compute_exponential_reward(height_error, scale=1.0, temperature=self.height_similarity_coeff), -1)
        
        if self.lin_vel_reward_mode == "linear":
            lin_vel_reward = torch.sum(compute_linear_reward(lin_vel_error, scale=1.0), -1)
        elif self.lin_vel_reward_mode == "square":
            lin_vel_reward = torch.sum(compute_square_reward(lin_vel_error, scale=1.0), -1)
        elif self.lin_vel_reward_mode == "exponential":
            lin_vel_reward = torch.sum(compute_exponential_reward(lin_vel_error, scale=1.0, temperature=self.lin_vel_similarity_coeff), -1)
        
        if self.ang_vel_reward_mode == "linear":
            ang_vel_reward = torch.sum(compute_linear_reward(ang_vel_error, scale=1.0), -1)
        elif self.ang_vel_reward_mode == "square":
            ang_vel_reward = torch.sum(compute_square_reward(ang_vel_error, scale=1.0), -1)
        elif self.ang_vel_reward_mode == "exponential":
            ang_vel_reward = torch.sum(compute_exponential_reward(ang_vel_error, scale=1.0, temperature=self.ang_vel_similarity_coeff), -1)
        
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
        assert self.position_reward_mode in ["linear", "square", "exponential"]
        assert self.yaw_reward_mode in ["linear", "square", "exponential"]
    
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
        
        if self.yaw_reward_mode == "linear":
            yaw_reward = torch.sum(compute_linear_reward(yaw_error, scale=1.0), -1)
        elif self.yaw_reward_mode == "square":
            yaw_reward = torch.sum(compute_square_reward(yaw_error, scale=1.0), -1)
        elif self.yaw_reward_mode == "exponential":
            yaw_reward = torch.sum(compute_exponential_reward(yaw_error, scale=1.0, temperature=self.yaw_coeff), -1)
        
        position_reward = self.position_weight*position_reward
        yaw_reward = self.yaw_weight*yaw_reward
        
        return position_reward, yaw_reward
    

@dataclass
class GoalTrackingReward:
    height_similarity_weight: float = 1.0
    goal_pos_dist_weight: float = 1.0
    goal_yaw_dist_weight: float = 1.0
    height_reward_mode:str="square"
    goal_pos_dist_reward_mode:str="square"
    goal_yaw_dist_reward_mode:str="square"
    
    def __post_init__(self):
        assert self.height_reward_mode in ["linear", "square", "exponential"]
        assert self.goal_pos_dist_reward_mode in ["linear", "square", "exponential"]
        assert self.goal_yaw_dist_reward_mode in ["linear", "square", "exponential"]
    
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
        
        if self.goal_pos_dist_reward_mode == "linear":
            goal_pos_dist_reward = torch.sum(compute_linear_reward(goal_pos_error, scale=1.0), -1)
        elif self.goal_pos_dist_reward_mode == "square":
            goal_pos_dist_reward = torch.sum(compute_square_reward(goal_pos_error, scale=1.0), -1)
        elif self.goal_pos_dist_reward_mode == "exponential":
            goal_pos_dist_reward = torch.sum(compute_exponential_reward(goal_pos_error, scale=1.0, temperature=0.05), -1)
        
        if self.goal_yaw_dist_reward_mode == "linear":
            goal_yaw_dist_reward = torch.sum(compute_linear_reward(goal_yaw_error, scale=1.0), -1)
        elif self.goal_yaw_dist_reward_mode == "square":
            goal_yaw_dist_reward = torch.sum(compute_square_reward(goal_yaw_error, scale=1.0), -1)
        elif self.goal_yaw_dist_reward_mode == "exponential":
            goal_yaw_dist_reward = torch.sum(compute_exponential_reward(goal_yaw_error, scale=1.0, temperature=0.05), -1)
        
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
        assert self.contact_reward_mode in ["linear", "square", "exponential"]
    
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
        
        weight = self.contact_similarity_weight * torch.ones_like(contact_error)
        contact_reward = weight * contact_reward
        
        return contact_reward


if __name__ == "__main__":
    position_x = torch.arange(0.0, 3.0, 0.1)
    position_y = torch.arange(-0.5, 0.5, 0.1)
    X, Y = torch.meshgrid(position_x, position_y, indexing="ij")
    
    goal_position = torch.tensor([3.0, 0.0]).view(1, -1)
    goal_yaw = torch.tensor([0.0]).view(1, -1)
    
    position = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)
    root_height = 0.55 * torch.ones(position.shape[0], 1)
    yaw = 3.14/6 * torch.ones(position.shape[0], 1)
    ankel_pitch = torch.zeros(position.shape[0], 2)
    reference_height = 0.55
    
    goal_pos_dist = goal_position - position
    goal_yaw_dist = goal_yaw - yaw
    
    goal_tracking_reward = GoalTrackingReward(
        height_similarity_weight= 0.0,
        goal_pos_dist_weight= 1.00,
        goal_yaw_dist_weight= 0.0,
        height_reward_mode="square",
        goal_pos_dist_reward_mode="square",
        goal_yaw_dist_reward_mode="exponential"
    )
    
    height_reward, goal_pos_dist_reward, goal_yaw_dist_reward = \
        goal_tracking_reward.compute_reward(root_height, goal_pos_dist, goal_yaw_dist, reference_height)
    
    
    height_reward_grid = height_reward.reshape(X.shape)
    goal_pos_dist_reward_grid = goal_pos_dist_reward.reshape(X.shape)
    goal_yaw_dist_reward_grid = goal_yaw_dist_reward.reshape(X.shape)
    
    total_reward_grid = height_reward_grid + goal_pos_dist_reward_grid + goal_yaw_dist_reward_grid
    
    import matplotlib.pyplot as plt
    plt.figure()
    # plt.imshow(height_reward_grid, cmap="jet", origin="lower")
    # plt.imshow(goal_pos_dist_reward_grid, cmap="jet", origin="lower")
    # plt.imshow(goal_yaw_dist_reward_grid, cmap="jet", origin="lower")
    # plt.imshow(ankle_pitch_reward_grid, cmap="jet",origin="lower")
    plt.imshow(total_reward_grid, cmap="jet", origin="lower")
    plt.colorbar()
    plt.xlabel("Y")
    plt.ylabel("X")
    plt.show()