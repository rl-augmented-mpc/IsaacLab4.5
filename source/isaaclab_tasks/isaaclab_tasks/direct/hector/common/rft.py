import torch
from dataclasses import dataclass
from typing import Optional

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
                 dynamic_friction_coef: list[float]=[0.5, 0.5], 
                 dumping_coef: list[float]=[10, 50, 100]):
        """
        Resistive Force Theory based soft terrain model https://www.science.org/doi/10.1126/science.1229163
        with Exponential Moving Average filter suggested in https://www.science.org/doi/10.1126/scirobotics.ade2256
        For implementation details, refer to supplementary material
        
        Args:
        - cfg (MaterialCfg): Soil parameters
        - num_envs (int): Number of environments
        - num_leg (int): Number of legs
        - num_contact_points (int): Number of contact points per leg
        - device (torch.device): Device
        - surface_area (float): Surface area of the intruder (i.e. foot sole)
        - static_friction_coef (list[float, float]): Static friction coefficient
        - dynamic_friction_coef (list[float, float]): Dynamic friction coefficient
        - dumping_coef (float): Dumping coefficient applied to normal direction (f_dumping = -kp * vz)
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_leg = num_leg
        self.num_contact_points = num_contact_points
        self.surface_area = surface_area
        self.dynamic_friction_coef = dynamic_friction_coef
        self.damping_coef = dumping_coef
        
        self.v_eps = 1e-3
        self.dA = self.surface_area/self.num_contact_points
        self.ground_height = 0.0
        self.max_depth = 0.03
        
        # body quantity history
        self.velocity_prev = torch.zeros((num_envs, num_leg*num_contact_points, 3), device=device)
        self.force_gm = torch.zeros((num_envs, num_leg*num_contact_points, 3), device=device)
        self.force_ema = torch.zeros((num_envs, num_leg*num_contact_points, 3), device=device)
        self.tau_r = torch.zeros((num_envs, num_leg*num_contact_points, 3), device=device)
        self.depth_mask = torch.zeros((num_envs, num_leg*num_contact_points), dtype=torch.long, device=device).bool()
        self.c_r = 12.5/500 # 100/f (original paper 100/4000)
    
    # def get_force(self, foot_depth:torch.Tensor, foot_rot: torch.Tensor, foot_velocity:torch.Tensor, beta:torch.Tensor, gamma:torch.Tensor, gait_contact:torch.Tensor)->torch.Tensor:
    #     """
    #     Find resistive force per foot
    #     """
    #     force = self.compute_resistive_force(foot_depth, foot_rot, foot_velocity, beta, gamma)
    #     # print(180/torch.pi * gamma)
        
    #     # Only enable RFT for stance leg
    #     left_rft_force = force[:, :self.num_contact_points, :]
    #     right_rft_force = force[:, self.num_contact_points:, :]

    #     left_gait_stance = gait_contact[:, :1, None].repeat(1, self.num_contact_points, 3).long()
    #     right_gait_stance = gait_contact[:, 1:, None].repeat(1, self.num_contact_points, 3).long()

    #     left_stance_contact_state = (foot_depth[:, :self.num_contact_points] < self.ground_height - 0.0)[:, :, None].long()
    #     left_swing_contact_state = (foot_depth[:, :self.num_contact_points] < self.ground_height - 0.02)[:, :, None].long()
    #     right_stance_contact_state = (foot_depth[:, self.num_contact_points:] < self.ground_height - 0.01)[:, :, None].long()
    #     right_swing_contact_state = (foot_depth[:, self.num_contact_points:] < self.ground_height - 0.0)[:, :, None].long()

    #     left_rft_force[left_gait_stance.bool()] = (left_rft_force * left_stance_contact_state).reshape(-1, self.num_contact_points, 3)[left_gait_stance.bool()]
    #     left_rft_force[(1-left_gait_stance).bool()] = (left_rft_force * left_swing_contact_state).reshape(-1, self.num_contact_points, 3)[(1-left_gait_stance).bool()]
    #     right_rft_force[right_gait_stance.bool()] = (right_rft_force * right_stance_contact_state).reshape(-1, self.num_contact_points, 3)[right_gait_stance.bool()]
    #     right_rft_force[(1-right_gait_stance).bool()] = (right_rft_force * right_swing_contact_state).reshape(-1, self.num_contact_points, 3)[(1-right_gait_stance).bool()]
        
    #     filtered_force = torch.cat((left_rft_force, right_rft_force), dim=1)
    #     return filtered_force

    def get_force(self, foot_depth:torch.Tensor, foot_rot: torch.Tensor, foot_velocity:torch.Tensor, beta:torch.Tensor, gamma:torch.Tensor, gait_contact:torch.Tensor)->torch.Tensor:
        """
        Find resistive force per foot
        """
        force = self.compute_resistive_force(foot_depth, foot_rot, foot_velocity, beta, gamma)
        filtered_force = self.filter_with_contact(foot_depth, foot_velocity, force)
        return filtered_force
    
    def filter_with_contact(self, foot_depth:torch.Tensor, foot_velocity:torch.Tensor, force:torch.Tensor)->torch.Tensor:
        """
        Filter force with contact state
        """
        contact_mask = foot_depth < self.ground_height
        force = force * contact_mask[:, :, None]
        return force
    
    def compute_resistive_force(self, foot_depth:torch.Tensor, foot_rot:torch.Tensor, foot_velocity:torch.Tensor, beta:torch.Tensor, gamma:torch.Tensor):
        """
        Find resistive force per foot
        """
        # from world frame to local frame
        foot_velocity = torch.bmm(foot_rot.reshape(-1, 3, 3).transpose(1,2), foot_velocity.reshape(-1, 3, 1)).squeeze(-1).reshape(-1, self.num_leg*self.num_contact_points, 3)

        # Get resistive force of elementary parts
        alpha_x, alpha_z = self.compute_elementary_force(beta, gamma)
        
        # RFT works only when foot is penetrating
        depth_mask = foot_depth < self.ground_height # apply resistive force only when foot is penetrating
        
        # Normal force
        fz = alpha_z * (-foot_depth) * self.dA * depth_mask * (1e6) #m^3 to mm^3
        # Sagittal force
        fx = -alpha_x * (-foot_depth) * self.dA * depth_mask * (1e6) #m^3 to mm^3 (goes against the saggital velocity)
        fx = fx * torch.sign(foot_velocity[:, :, 0])
        # Lateral force
        fy = torch.zeros_like(fx)

        self.force_gm = torch.stack((fx, fy, fz), dim=-1)
        self.emf_filteing(foot_velocity, foot_depth)
        
        # from global to local frame
        # self.force_ema = torch.bmm(foot_rot.reshape(-1, 3, 3).transpose(1,2), self.force_ema.reshape(-1, 3, 1)).squeeze(-1).reshape(-1, self.num_leg*self.num_contact_points, 3)
        
        # add damping term
        self.force_ema[:, :, 0] += -self.damping_coef[0]* foot_velocity[:, :, 0] * depth_mask # damping term
        self.force_ema[:, :, 1] += -self.damping_coef[1] * foot_velocity[:, :, 1] * depth_mask # damping term
        self.force_ema[:, :, 2] += -self.damping_coef[2] * foot_velocity[:, :, -1] * depth_mask # damping term
        
        return self.force_ema
    
    def compute_elementary_force(self, beta:torch.Tensor, gamma:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        """
        Compute elementary force per foot
        """
        # Use beta=0, gamma=pi/2 for alpha_z
        # gamma_z = torch.pi/2 * torch.ones_like(gamma) # always pi/2
        # beta_z = torch.zeros_like(beta) # always 0
        # alpha_z = torch.zeros_like(beta) # A00
        # alpha_z += self.cfg.A00*torch.cos(2*torch.pi*(0*beta_z/torch.pi)) #A00
        # alpha_z += self.cfg.A10*torch.cos(2*torch.pi*(1*beta_z/torch.pi)) # A10
        # alpha_z += self.cfg.B01*torch.sin(2*torch.pi*(1*gamma_z/(2*torch.pi))) # B01
        # alpha_z += self.cfg.B11*torch.sin(2*torch.pi*(1*beta_z/torch.pi + 1*gamma_z/(2*torch.pi))) # B11
        # alpha_z += self.cfg.B_11*torch.sin(2*torch.pi*(-1*beta_z/torch.pi + 1*gamma_z/(2*torch.pi))) # B-11
        
        alpha_z = torch.zeros_like(beta) # A00
        alpha_z += self.cfg.A00*torch.cos(2*torch.pi*(0*beta/torch.pi)) #A00
        alpha_z += self.cfg.A10*torch.cos(2*torch.pi*(1*beta/torch.pi)) # A10
        alpha_z += self.cfg.B01*torch.sin(2*torch.pi*(1*gamma/(2*torch.pi))) # B01
        alpha_z += self.cfg.B11*torch.sin(2*torch.pi*(1*beta/torch.pi + 1*gamma/(2*torch.pi))) # B11
        alpha_z += self.cfg.B_11*torch.sin(2*torch.pi*(-1*beta/torch.pi + 1*gamma/(2*torch.pi))) # B-11
        
        # calculate alpha_x
        alpha_x = torch.zeros_like(gamma)
        alpha_x += self.cfg.C01*torch.sin(2*torch.pi*(1*gamma/(2*torch.pi))) # C01
        alpha_x += self.cfg.C11*torch.sin(2*torch.pi*(1*beta/(torch.pi) + 1*gamma/(2*torch.pi))) # C11
        alpha_x += self.cfg.C_11*torch.sin(2*torch.pi*(-1*beta/(torch.pi) + 1*gamma/(2*torch.pi))) # C-11
        alpha_x += self.cfg.D10*torch.cos(2*torch.pi*(1*beta/(torch.pi))) # D10
        
        return alpha_x, alpha_z
    
    def emf_filteing(self, velocity:torch.Tensor, depth:torch.Tensor):
        """
        Exponential Moving Average Filtering descirbed in https://www.science.org/doi/10.1126/scirobotics.ade2256
        """
        increment_mask = velocity*self.velocity_prev < 0
        tau_r_boundary = self.tau_r < 1
        self.depth_mask = (depth < self.ground_height-self.max_depth) | (self.depth_mask) # activate once foot reaches maximum depth
        mask = increment_mask & tau_r_boundary & self.depth_mask[:, :, None]
        self.tau_r[mask] += self.c_r

        # apply EMA filter
        self.force_ema = (1-0.8*self.tau_r)*self.force_gm + 0.8*self.tau_r*self.force_ema
        self.velocity_prev = velocity


if __name__ == "__main__":
    cfg = PoppySeedLPCfg()
    # cfg = PoppySeedCPCfg()
    rft = RFT_EMF(cfg, torch.device("cuda"), 1, 2, 1, 0.1*0.15)
    
    foot_depth = 0.02 * torch.ones((1, 2), device=torch.device("cuda"))
    foot_velocity = torch.zeros((1, 2, 3), device=torch.device("cuda"))
    beta = torch.zeros((1, 2), device=torch.device("cuda"))
    gamma = torch.pi/2 * torch.ones((1, 2), device=torch.device("cuda"))
    stance_state = torch.ones((1, 2), device=torch.device("cuda"))
    
    alpha_x, alpha_z = rft.compute_elementary_force(beta, gamma)
    print(alpha_z)