import torch
from dataclasses import dataclass
from typing import Optional, Sequence

@dataclass
class MaterialCfg:
    A00: Optional[float] = None
    A10: Optional[float] = None
    B11: Optional[float] = None
    B01: Optional[float] = None
    B_11: Optional[float] = None
    C11: Optional[float] = None
    C01: Optional[float] = None
    C_11: Optional[float] = None
    D10: Optional[float] = None

@dataclass
class PoppySeedLPCfg(MaterialCfg):
    A00: float = 0.051
    A10: float = 0.047
    B11: float = 0.053
    B01: float = 0.083
    B_11: float = 0.020
    C11: float = -0.026
    C01: float = 0.057
    C_11: float = 0.0
    D10: float = 0.025

@dataclass
class PoppySeedCPCfg(MaterialCfg):
    A00: float = 0.094
    A10: float = 0.092
    B11: float = 0.092
    B01: float = 0.151
    B_11: float = 0.035
    C11: float = -0.039
    C01: float = 0.086
    C_11: float = 0.018
    D10: float = 0.046

class RFT_EMF:
    def __init__(self, 
                 cfg: MaterialCfg, 
                 device: torch.device|str,
                 num_envs: int,
                 num_leg: int,
                 num_contact_points: int,
                 surface_area: float,
                 damping_coef: torch.Tensor, 
                 dynamic_friction_coef: torch.Tensor, 
                 compactness: torch.Tensor):
        """
        Resistive Force Theory based soft terrain model https://www.science.org/doi/10.1126/science.1229163
        with Exponential Moving Average filter suggested in https://www.science.org/doi/10.1126/scirobotics.ade2256
        For implementation details, refer to supplementary material
        
        Args:
        - cfg (MaterialCfg): Soil parameters
        - device (torch.device): Device
        - num_envs (int): Number of environments
        - num_leg (int): Number of legs
        - num_contact_points (int): Number of contact points per leg
        - surface_area (float): Surface area of the intruder (i.e. foot sole)
        - dumping_coef (float): Dumping coefficient applied to normal direction (f_dumping = -kp * vz)
        - dynamic_friction_coef (list[float, float]): Dynamic friction coefficient
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_leg = num_leg
        self.num_contact_points = num_contact_points
        self.surface_area = surface_area
        
        assert damping_coef.shape[1] == 3, "Need x,y,z damping coefficient"
        self.damping_coef = damping_coef
        self.dynamic_friction_coef = dynamic_friction_coef
        self.compactness = compactness
        
        self.v_eps = 1e-3
        self.dA = self.surface_area/self.num_contact_points
        self.ground_height = 0.0
        self.max_depth = 0.2
        
        # body quantity history
        self.velocity_prev = torch.zeros((num_envs, num_leg*num_contact_points, 3), device=device)
        self.force_gm = torch.zeros((num_envs, num_leg*num_contact_points, 3), device=device)
        self.force_ema = torch.zeros((num_envs, num_leg*num_contact_points, 3), device=device)
        self.force_damping = torch.zeros((num_envs, num_leg*num_contact_points, 3), device=device)
        self.tau_r = torch.zeros((num_envs, num_leg*num_contact_points, 3), device=device)
        self.depth_mask = torch.zeros((num_envs, num_leg*num_contact_points), dtype=torch.long, device=device).bool()
        self.c_r = 12.5/500 # 100/f (original paper 100/4000)
        
    def set_damping_coef(self, damping_coef:torch.Tensor, env_id:Sequence[int])->None:
        """
        Change damping coefficient
        """
        self.damping_coef[env_id] = damping_coef
    
    def set_dynamic_friction_coef(self, dynamic_friction_coef:torch.Tensor, env_id:Sequence[int])->None:
        """
        Change dynamic friction coefficient
        """
        self.dynamic_friction_coef[env_id] = dynamic_friction_coef
    
    def set_soil_compactness(self, compactness:torch.Tensor, env_id:Sequence[int])->None:
        """
        Change compactness
        """
        self.compactness[env_id] = compactness

    def get_force(self, foot_pos_z:torch.Tensor, foot_rot: torch.Tensor, foot_velocity:torch.Tensor, beta:torch.Tensor, gamma:torch.Tensor, gait_contact:torch.Tensor)->torch.Tensor:
        """
        Find resistive force per foot element
        """
        # foot_depth = (self.ground_height - foot_pos_z) * (foot_pos_z < self.ground_height)
        foot_depth = (self.ground_height - foot_pos_z + 0.01) * (foot_pos_z < self.ground_height - 0.01)
        self.compute_resistive_force(foot_depth, foot_rot, foot_velocity, beta, gamma)
        # filter swing leg
        # filtered_damping_force = self.filter_swing_leg(self.force_damping, gait_contact)
        # return self.force_ema + filtered_damping_force
        return self.force_ema + self.force_damping
    
    def filter_swing_leg(self, force:torch.Tensor, gait_contact:torch.Tensor)->torch.Tensor:
        """
        Filter force of swing leg. 
        Do not apply x damping to swing leg
        """
        contact_points = force.shape[1]
        swing_mask = 1 - gait_contact
        left_swing_mask = swing_mask[:, 0][:, None].repeat(1, contact_points//2)
        right_swing_mask = swing_mask[:, 1][:, None].repeat(1, contact_points//2)
        swing_mask_long = torch.cat((left_swing_mask, right_swing_mask), dim=1).unsqueeze(2).repeat(1,1,3).bool() # (num_envs, num_contact_points, 3)
        force[swing_mask_long[:, :, 0]] = force[swing_mask_long[:, :, 0]] * 0.0
        return force
    
    def compute_resistive_force(self, foot_depth:torch.Tensor, foot_rot:torch.Tensor, foot_velocity:torch.Tensor, beta:torch.Tensor, gamma:torch.Tensor):
        """
        Find resistive force per foot
        """
        # transform velocity in world frame to foot local frame
        foot_velocity_b = torch.bmm(foot_rot.reshape(-1, 3, 3).transpose(1,2), foot_velocity.reshape(-1, 3, 1)).squeeze(-1).reshape(-1, self.num_leg*self.num_contact_points, 3)

        # get resistive force of elementary parts
        alpha_x, alpha_z = self.compute_elementary_force(beta, gamma)
        
        # Normal force
        fz = alpha_z * foot_depth * self.dA * (1e6) #m^3 to mm^3
        # Sagittal force
        fx = alpha_x * foot_depth * self.dA * (1e6) #m^3 to mm^3 (goes against the saggital velocity)
        fx = -fx * torch.sign(foot_velocity_b[:, :, 0])
        fy = torch.zeros_like(fx)
        self.force_gm = torch.stack((fx, fy, fz), dim=-1) # (num_envs, num_contact_points, 3)
        
        # RFT works only when foot is penetrating
        depth_mask = (foot_depth > 0).unsqueeze(2) # (num_envs, num_contact_points, 1)
        penetration_mask = (gamma > 0).unsqueeze(2)  # (num_envs, num_contact_points, 1)
        self.force_gm = self.force_gm * depth_mask * penetration_mask
        
        # exponential moving average filtering
        self.emf_filteing(foot_velocity_b, foot_depth)
        
        # from global to local frame (with beta nearly 0, you dont really need this)
        fx = self.force_ema[:, :, 0]
        fz = self.force_ema[:, :, 2]
        self.force_ema[:, :, 0] = fx*torch.cos(beta) - fz*torch.sin(beta)
        self.force_ema[:, :, 2] = fx*torch.sin(beta) + fz*torch.cos(beta)
        
        # filter with velocity and depth again
        self.force_ema = self.force_ema * depth_mask * penetration_mask
        
        # compute damping force woth coulomb friction model
        self.compute_damping_force(foot_depth, foot_velocity_b, gamma)
    
    
    def compute_damping_force(self, foot_depth:torch.Tensor, foot_velocity_b:torch.Tensor, gamma:torch.Tensor)->None:
        """
        Compute damping force
        
        foot_depth: (num_envs, num_contact_points)
        foot_velocity_b: (num_envs, num_contact_points, 3)
        normal_force: (num_envs, num_contact_points)
        """
        
        v_tangential = foot_velocity_b[:, :, :2]
        v_tangential_norm = torch.norm(v_tangential, dim=-1, keepdim=True) # (num_envs, num_contact_points, 1)
        
        # This is like a spring trying to bring v_t to zero.
        force_tangential_candidate = -self.damping_coef[:, :2].unsqueeze(1) * v_tangential + self.force_ema[:, :, 0:2] # (num_envs, num_contact_points, 2)
        force_tangential_candidate_norm = torch.norm(force_tangential_candidate, dim=-1, keepdim=True) # (num_envs, num_contact_points, 1)

        # determine the maximum allowable friction force magnitude (Coulomb limit).
        force_tangential_max = self.dynamic_friction_coef[:, None, None] * self.force_ema[:, :, 2:]  # (num_envs, num_contact_points, 1)

        # compute friction force
        eps = 1e-6
        force_tangential_sliding = -self.dynamic_friction_coef[:, None, None] * self.force_ema[:, :, 2:] * (v_tangential / (v_tangential_norm + eps)) # (num_envs, num_contact_points, 2)

        # friction cone check
        force_tangential = torch.where(force_tangential_candidate_norm <= force_tangential_max, force_tangential_candidate - self.force_ema[:, :, 0:2], force_tangential_sliding)
        
        self.force_damping[:, :, :2] = force_tangential # x,y damping
        self.force_damping[:, :, 2] = -self.damping_coef[:, 2].unsqueeze(1) * foot_velocity_b[:, :, 2] # z damping
        
        depth_mask = foot_depth > 0
        penetration_mask = gamma > 0
        self.force_damping = self.force_damping * depth_mask[:, :, None] * penetration_mask[:, :, None]
    
    def compute_elementary_force(self, beta:torch.Tensor, gamma:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        """
        Compute elementary force per foot per depth
        """
        alpha_z = torch.zeros_like(beta) # A00
        alpha_z += self.cfg.A00*torch.cos(2*torch.pi*(0*beta/torch.pi)) #A00
        alpha_z += self.cfg.A10*torch.cos(2*torch.pi*(1*beta/torch.pi)) # A10
        alpha_z += self.cfg.B01*torch.sin(2*torch.pi*(1*gamma/(2*torch.pi))) # B01
        alpha_z += self.cfg.B11*torch.sin(2*torch.pi*(1*beta/torch.pi + 1*gamma/(2*torch.pi))) # B11
        alpha_z += self.cfg.B_11*torch.sin(2*torch.pi*(-1*beta/torch.pi + 1*gamma/(2*torch.pi))) # B-11
        
        # calculate alpha_x
        alpha_x = torch.zeros_like(gamma)
        alpha_x += self.cfg.C01*torch.cos(2*torch.pi*(1*gamma/(2*torch.pi))) # C01
        alpha_x += self.cfg.C11*torch.cos(2*torch.pi*(1*beta/(torch.pi) + 1*gamma/(2*torch.pi))) # C11
        alpha_x += self.cfg.C_11*torch.cos(2*torch.pi*(-1*beta/(torch.pi) + 1*gamma/(2*torch.pi))) # C-11
        alpha_x += self.cfg.D10*torch.sin(2*torch.pi*(1*beta/(torch.pi))) # D10
        
        alpha_z *= self.compactness.unsqueeze(1)
        alpha_x *= self.compactness.unsqueeze(1)
        
        return alpha_x, alpha_z
    
    def emf_filteing(self, velocity:torch.Tensor, depth:torch.Tensor):
        """
        Exponential Moving Average Filtering descirbed in https://www.science.org/doi/10.1126/scirobotics.ade2256
        """
        increment_mask = velocity * self.velocity_prev < 0 # (num_envs, num_leg*num_contact_points, 3)
        tau_r_boundary = self.tau_r < 1 # (num_envs, num_leg*num_contact_points, 3)
        self.depth_mask = (depth > self.max_depth) | (self.depth_mask) # activate once foot reaches maximum depth
        mask = increment_mask & tau_r_boundary & self.depth_mask[:, :, None] # (num_envs, num_leg*num_contact_points, 3)
        self.tau_r[mask] += self.c_r
        # reset tau_r when foot gets outside granular media
        non_contact_mask = (depth > self.ground_height)
        self.tau_r[non_contact_mask] = 0.0

        # apply EMA filter
        self.force_ema = (1-0.8*self.tau_r)*self.force_gm + 0.8*self.tau_r*self.force_ema
        # self.force_ema[:, :, 0] = self.force_gm[:, :, 0] # ignore x filtering
        self.velocity_prev = velocity
    
    # # fixed filtering
    # def emf_filteing(self, velocity:torch.Tensor, depth:torch.Tensor):
    #     """
    #     Exponential Moving Average Filter with fixed alpha = 0.972
    #     """
    #     self.force_ema = (1-0.8)*self.force_gm + 0.8*self.force_ema
    #     self.velocity_prev = velocity
    
    # # null filtering
    # def emf_filteing(self, velocity:torch.Tensor, depth:torch.Tensor):
    #     """
    #     No filtering
    #     """
    #     self.force_ema = self.force_gm
    #     self.velocity_prev = velocity


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    cfg = PoppySeedCPCfg()
    # cfg = PoppySeedLPCfg()
    
    rft = RFT_EMF(
        cfg=cfg, 
        device="cuda",
        num_envs=1,
        num_leg=2, 
        num_contact_points=1, 
        surface_area=0.01*0.015,
        damping_coef=torch.tensor([0.1, 0.1, 0.1], device="cuda").unsqueeze(0),
        dynamic_friction_coef=torch.tensor([0.8], device="cuda").unsqueeze(0), 
        compactness=torch.tensor([1.0], device="cuda").unsqueeze(0)
    )
    
    num_bin = 50
    beta = torch.linspace(-torch.pi/2, torch.pi/2, num_bin, device="cuda")
    gamma = torch.linspace(-torch.pi/2, torch.pi/2, num_bin, device="cuda")
    XX, YY = torch.meshgrid(beta, gamma, indexing="ij")
    beta = XX.flatten()
    gamma = YY.flatten()
    print(beta.shape, gamma.shape)
    
    alpha_x, alpha_z = rft.compute_elementary_force(beta.unsqueeze(0), gamma.unsqueeze(0))
    
    
    beta = beta.cpu().numpy().flatten()
    gamma = gamma.cpu().numpy().flatten()
    alpha_x = alpha_x.cpu().numpy().flatten()
    alpha_z = alpha_z.cpu().numpy().flatten()
    
    alpha_x_pixel = alpha_x.reshape(num_bin, num_bin)
    alpha_z_pixel = alpha_z.reshape(num_bin, num_bin)
    
    alpha_x_pixel = np.flipud(alpha_x_pixel)
    alpha_z_pixel = np.flipud(alpha_z_pixel)
        
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(alpha_z_pixel, cmap="jet", vmin=-np.max(np.abs(alpha_z)), vmax=np.max(np.abs(alpha_z)))
    plt.xticks([0, num_bin//2, num_bin-1], [r"-$\pi$/2", "0", r"$\pi$/2"])
    plt.yticks([0, num_bin//2, num_bin-1], [r"$\pi$/2", "0", r"-$\pi$/2"])
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\gamma$")
    plt.colorbar()
    plt.title("Alpha Z")
    
    plt.subplot(1, 2, 2)
    plt.imshow(alpha_x_pixel, cmap="jet", vmin=-np.max(np.abs(alpha_x)), vmax=np.max(np.abs(alpha_x)))
    plt.xticks([0, num_bin//2, num_bin-1], [r"-$\pi$/2", "0", r"$\pi$/2"])
    plt.yticks([0, num_bin//2, num_bin-1], [r"$\pi$/2", "0", r"-$\pi$/2"])
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\gamma$")
    plt.colorbar()
    plt.title("Alpha X")
    plt.show()