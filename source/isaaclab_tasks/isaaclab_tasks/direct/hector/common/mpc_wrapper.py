from multiprocessing.shared_memory import SharedMemory
from dataclasses import dataclass
import numpy as np

from hector_control import HectorController

@dataclass
class Robot_Conf:
    mass: float = 11.578
    I_c: np.ndarray = np.array(
        [[0.5413, 0.0, 0.0], 
         [0.0, 0.5200, 0.0], 
         [0.0, 0.0, 0.0691]], dtype=np.float32)

@dataclass
class MPC_Conf:
    """
    MPC Configuration
    
    control_dt (float): control timestep in sec
    control_iteration_between_mpc (int): number of control iterations between MPC updates
    horizon_length (int): MPC horizon length
    dsp_durations (np.ndarray): double support phase durations (2,)
    ssp_durations (np.ndarray): single support phase durations (2,)
    """
    control_dt: float
    control_iteration_between_mpc: int
    horizon_length: int
    mpc_decimation: int
    dsp_durations: np.ndarray
    ssp_durations: np.ndarray


class MPCWrapper:
    """
    Wrapper around Hector MPC controller with shared memory management.
    """
    def __init__(self, mpc_conf, use_shared_memory=True):
        self.mpc_conf = mpc_conf
        self.robot_conf = Robot_Conf()
        self.use_shared_memory = use_shared_memory

        # Initialize shared memory for observation vector if required
        if self.use_shared_memory:
            self.shared_mem = SharedMemory(create=True, size=33 * np.float64().nbytes)
            self.obs_np = np.ndarray((33,), dtype=np.float64, buffer=self.shared_mem.buf)
        else:
            self.obs_np = np.zeros(33, dtype=np.float64)

        self.hc = HectorController(
            mpc_conf.control_dt,
            mpc_conf.control_iteration_between_mpc,
            mpc_conf.horizon_length,
            mpc_conf.mpc_decimation,
            mpc_conf.dsp_durations,
            mpc_conf.ssp_durations,
        )
        self.hc.setFrictionCoefficient(0.3)
    
    def reset(self)->None:
        self.hc.reset()

    def update_state(self, obs: np.ndarray) -> None:
        """
        Update the controller state.

        Args:
            obs (np.ndarray): observation vector (33,)
        """
        if self.use_shared_memory:
            np.copyto(self.obs_np, obs)
        else:
            self.obs_np[:] = obs

        self.hc.setState(self.obs_np[:3], self.obs_np[3:7], self.obs_np[7:10],
                         self.obs_np[10:13], self.obs_np[13:23], self.obs_np[23:])

    def set_command(self, gait_num: int, roll_pitch: np.ndarray, twist: np.ndarray, height: float) -> None:
        """
        Update the command.

        Args:
            gait_num (int): gait number
            roll_pitch (np.ndarray): (2,)
            twist (np.ndarray): (3,)
            height (float): target height
        """
        self.hc.setGaitNum(gait_num)
        self.hc.setTargetCommand(roll_pitch, twist, height)

    def set_swing_parameters(self, stepping_frequency: float, foot_height: float) -> None:
        """
        Set swing parameters.

        Args:
            stepping_frequency (float): gait stepping frequency
            foot_height (float): foot height
        """
        self.hc.setGaitSteppingFrequency(stepping_frequency)
        self.hc.setFootHeight(foot_height)

    def set_srbd_peturbation(self, accel: np.ndarray, ang_accel: np.ndarray) -> None:
        """
        Set perturbation to Single Rigid Body Dynamics.

        Args:
            accel (np.ndarray): acceleration (3,)
            ang_accel (np.ndarray): angular acceleration (3,)
        """
        self.hc.setSRBDPerturbation(accel, ang_accel)

    def set_srbd_residual(self, A_residual: np.ndarray, B_residual: np.ndarray) -> None:
        """
        Set SRBD residual.

        Args:
            A_residual (np.ndarray): A residual (13, 13)
            B_residual (np.ndarray): B residual (13, 12)
        """
        self.hc.setSRBDResidual(A_residual, B_residual)

    def run(self) -> None:
        """
        Run MPC and low-level controller.
        """
        self.hc.run()

    def add_grf_residual(self, residual: np.ndarray) -> None:
        """
        Add GRF residual.

        Args:
            residual (np.ndarray): GRF residual (12,) (left_grf, left_grm, right_grf, right_grm)
        """
        self.hc.addResidualGRFM(residual)

    def add_foot_placement_residual(self, residual: np.ndarray) -> None:
        """
        Add foot placement residual.

        Args:
            residual (np.ndarray): GRF residual (4,) (left_residual, right_residual)
        """
        self.hc.addResidualFootPlacement(residual)

    def add_joint_position_residual(self, residual: np.ndarray) -> None:
        """
        Add joint position residual.

        Args:
            residual (np.ndarray): joint position residual (10,)
        """
        self.hc.addResidualJointPosition(residual)

    def update_low_level_controller(self) -> None:
        """
        Update low-level controller.
        """
        self.hc.updateLowLevelCommand()

    def get_action(self) -> np.ndarray:
        """
        Get joint torques.

        Returns:
            np.ndarray: joint torques (10,)
        """
        self.hc.computeAction()
        return np.array(self.hc.getTorque(), dtype=np.float64)
    
    def accel_gyro(self, rot_mat:np.ndarray)->np.ndarray:
        grfm = self.grfm.reshape(12,1)
        accel = (grfm[:3]+grfm[6:9])/self.robot_conf.mass
        
        # This is not correct (I_world * gyro = grm + r x grf)
        # I_world = rot_mat.T @ self.robot_conf.I_c @ rot_mat
        # I_world_inv = np.linalg.inv(I_world)
        # gyro = I_world_inv @ (grfm[3:6]+grfm[9:12])
        gyro = np.zeros_like(accel)
        return np.concatenate([accel, gyro])

    @property
    def grfm(self) -> np.ndarray:
        return self.hc.getGRFM()

    @property
    def contact_phase(self) -> np.ndarray:
        return self.hc.getContactPhase()

    @property
    def swing_phase(self) -> np.ndarray:
        return self.hc.getSwingPhase()

    @property
    def contact_state(self) -> np.ndarray:
        return self.hc.getContactState()

    @property
    def swing_state(self) -> np.ndarray:
        return self.hc.getSwingState()

    @property
    def reibert_foot_placement(self) -> np.ndarray:
        return self.hc.getReibertFootPlacement()

    @property
    def foot_placement(self) -> np.ndarray:
        return self.hc.getFootPlacement()

    @property
    def ref_foot_pos(self) -> np.ndarray:
        return self.hc.getRefFootPosition()

    @property
    def foot_pos(self) -> np.ndarray:
        return self.hc.getFootPosition()

    def __del__(self):
        """
        Clean up shared memory.
        """
        if self.use_shared_memory:
            self.shared_mem.close()
            self.shared_mem.unlink()

# class MPCWrapper:
#     """
#     Wrapper around Hector MPC controller
#     """
#     def __init__(self, mpc_conf: MPC_Conf)->None:
#         self.mpc_conf = mpc_conf
#         self.robot_conf = Robot_Conf()
#         self.hc = HectorController(mpc_conf.control_dt, 
#                                    mpc_conf.control_iteration_between_mpc, 
#                                    mpc_conf.horizon_length, 
#                                    mpc_conf.mpc_decimation,
#                                    mpc_conf.dsp_durations,
#                                    mpc_conf.ssp_durations)
    
#     def reset(self, obs:np.ndarray)->None:
#         """
#         Reset the controller

#         Args:
#             obs (np.ndarray): observation vector (33,)
#         """
#         gait_num = 1 # standing
#         roll_pitch = np.array([0.0, 0.0], dtype=np.float32)
#         twist = np.array([0.0, 0.0, 0.0], dtype=np.float32)
#         self.hc.setGaitNum(gait_num)
#         self.hc.setTargetCommand(roll_pitch, twist)
#         self.hc.setState(obs[:3], obs[3:7], obs[7:10], obs[10:13], obs[13:13+10], obs[23:23+10])
    
#     def update_state(self, obs:np.ndarray)->None:
#         """
#         Update the controller state

#         Args:
#             obs (np.ndarray): observation vector (33,)
#         """
#         self.hc.setState(obs[:3], obs[3:7], obs[7:10], obs[10:13], obs[13:13+10], obs[23:23+10])
    
#     def set_command(self, gait_num:int, roll_pitch:np.ndarray, twist:np.ndarray, height:float)->None:
#         """
#         update the command

#         Args:
#             gait_num (int): gait number
#             roll_pitch (np.ndarray): (2,)
#             twist (np.ndarray): (3,)
#         """
#         self.hc.setGaitNum(gait_num)
#         self.hc.setTargetCommand(roll_pitch, twist, height)
    
#     def set_swing_parameters(self, stepping_frequency:float, foot_height:float)->None:
#         """
#         Set swing parameters

#         Args:
#             stepping_frequency (float): gait stepping frequency
#             foot_height (float): foot height
#         """
#         self.hc.setGaitSteppingFrequency(stepping_frequency)
#         self.hc.setFootHeight(foot_height)
    
#     def set_srbd_peturbation(self, accel:np.ndarray, ang_accel:np.ndarray)->None:
#         """
#         Set perturbation to Single Rigid Body Dynamics

#         Args:
#             accel (np.ndarray): acceleration (3,)
#             ang_accel (np.ndarray): angular acceleration (3,)
#         """
#         self.hc.setSRBDPerturbation(accel, ang_accel)
    
#     def set_srbd_residual(self, A_residual:np.ndarray, B_residual:np.ndarray)->None:
#         """
#         Set SRBD residual

#         Args:
#             A_residual (np.ndarray): A residual (13,13)
#             B_residual (np.ndarray): B residual (13,12)
#         """
#         self.hc.setSRBDResidual(A_residual, B_residual)
    
#     def run(self):
#         """
#         Run MPC and low level controller
#         """
#         self.hc.run()
    
#     def add_grf_residual(self, residual:np.ndarray)->None:
#         """
#         Add GRF residual
        
#         Args:
#             residual (np.ndarray): GRF residual (12,) (left_grf, left_grm, right_grf, right_grm)
#         """
#         self.hc.addResidualGRFM(residual)
    
#     def add_foot_placement_residual(self, residual:np.ndarray)->None:
#         """
#         Add foot placement residual
        
#         Args:
#             residual (np.ndarray): GRF residual (4,) (left_residual, right_residual)
#         """
#         self.hc.addResidualFootPlacement(residual)
    
#     def add_joint_position_residual(self, residual:np.ndarray)->None:
#         """
#         Add joint position residual
        
#         Args:
#             residual (np.ndarray): joint position residual (10,)
#         """
#         self.hc.addResidualJointPosition(residual)
    
#     def update_low_level_controller(self):
#         """
#         Update low level controller
#         """
#         self.hc.updateLowLevelCommand()
    
#     def get_action(self)->np.ndarray:
#         """
#         Get joint torques
        
#         return:
#             np.ndarray: joint torques (10,)
#         """
#         self.hc.computeAction()
#         action = self.hc.getTorque()
#         return action
    
#     def accel_gyro(self, rot_mat:np.ndarray)->np.ndarray:
#         grfm = self.grfm.reshape(12,1)
#         accel = rot_mat @ (grfm[:3]+grfm[6:9])/self.robot_conf.mass
#         # I_c = R @ I_c @ R.T (R = rot_mat.T)
#         I_world = rot_mat.T @ self.robot_conf.I_c @ rot_mat
#         I_world_inv = np.linalg.inv(I_world)
#         gyro = I_world_inv @ rot_mat @ (grfm[3:6]+grfm[9:12])
#         return np.concatenate([accel, gyro])
    
#     @property
#     def grfm(self)->np.ndarray:
#         """
#         Get GRFM
        
#         return:
#             np.ndarray: GRFM (12,)
#         """
#         return self.hc.getGRFM()
    
#     @property
#     def contact_phase(self)->np.ndarray:
#         """
#         Get contact phase (subphase of each contact cycle)
        
#         return:
#             np.ndarray: contact phase (2,)
#         """
#         return self.hc.getContactPhase()
    
#     @property
#     def swing_phase(self)->np.ndarray:
#         """
#         Get swing phase (subphase of each swing cycle)
        
#         return:
#             np.ndarray: swing phase (2,)
#         """
#         return self.hc.getSwingPhase()
    
#     @property
#     def contact_state(self)->np.ndarray:
#         """
#         Get contact state
#         0 for swing, 1 for stance
        
#         return:
#             np.ndarray: contact state (2,)
#         """
#         return self.hc.getContactState()
    
#     @property
#     def swing_state(self)->np.ndarray:
#         """
#         Get swing state
#         0 for stance, 1 for swing
        
#         return:
#             np.ndarray: swing state (2,)
#         """
#         return self.hc.getSwingState()
    
#     @property
#     def reibert_foot_placement(self)->np.ndarray:
#         """
#         Get 2d reibert foot placement for both legs
        
#         return:
#             np.ndarray: foot placement (4,)
#         """
#         return self.hc.getReibertFootPlacement()
    
#     @property
#     def foot_placement(self)->np.ndarray:
#         """
#         Get 2d foot placement for both legs
        
#         return:
#             np.ndarray: foot placement (4,)
#         """
#         return self.hc.getFootPlacement()
    
#     @property
#     def ref_foot_pos(self)->np.ndarray:
#         """
#         Get reference foot position
        
#         return:
#             np.ndarray: reference foot position (6,)
#         """
#         return self.hc.getRefFootPosition()
    
#     @property
#     def foot_pos(self)->np.ndarray:
#         """
#         Get foot position
        
#         return:
#             np.ndarray: foot position (6,)
#         """
#         return self.hc.getFootPosition