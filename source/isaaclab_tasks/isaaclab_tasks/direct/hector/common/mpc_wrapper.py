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
            mpc_conf.mpc_decimation
        )
        self.hc.setFrictionCoefficient(0.3)
    
    def set_planner(self, planner_name: str) -> None:
        self.hc.setFootPlacementPlanner(planner_name)
    
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
    
    def update_gait_parameter(self, dsp_durations: np.ndarray, ssp_durations: np.ndarray) -> None:
        """
        Update gait parameters.

        Args:
            dsp_durations (np.ndarray): double support phase durations (2,)
            ssp_durations (np.ndarray): single support phase durations (2,)
        """
        self.hc.updateGaitParameter(dsp_durations, ssp_durations)

    def set_swing_parameters(self, stepping_frequency: float, foot_height: float) -> None:
        """
        Set swing parameters.

        Args:
            stepping_frequency (float): gait stepping frequency
            foot_height (float): foot height
        """
        self.hc.setGaitSteppingFrequency(stepping_frequency)
        self.hc.setFootHeight(foot_height)
    
    def set_terrain_slope(self, slope: float) -> None:
        """
        Set terrain slope.

        Args:
            slope (float): terrain slope
        """
        self.hc.setSlopePitch(slope)

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
        gyro = np.linalg.inv(self.robot_conf.I_c) @ (grfm[3:6]+grfm[9:12])
        # gyro = np.zeros_like(accel)
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
    def ref_foot_pos_b(self) -> np.ndarray:
        return self.hc.getRefFootPosition()

    @property
    def foot_pos_b(self) -> np.ndarray:
        return self.hc.getFootPosition()

    def __del__(self):
        """
        Clean up shared memory.
        """
        if self.use_shared_memory:
            self.shared_mem.close()
            self.shared_mem.unlink()