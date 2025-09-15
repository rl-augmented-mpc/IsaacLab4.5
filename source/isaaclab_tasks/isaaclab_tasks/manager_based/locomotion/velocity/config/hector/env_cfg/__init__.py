from .action_cfg import (
    HECTORBlindLocomotionActionsCfg, 
    HECTORPerceptiveLocomotionActionsCfg, 
    HECTORSlipActionsCfg, 
    HECTORGPUBlindLocomotionActionsCfg, 
    HECTORGPUSlipActionsCfg,
    # HECTORL2TActionsCfg,
    HECTORRLBlindLocomotionActionsCfg,
)
from .command_cfg import HECTORCommandsCfg
from .curriculum_cfg import HECTORCurriculumCfg, HECTORSlipCurriculumCfg
from .event_cfg import HECTOREventCfg, HECTORSlipEventCfg
from .observation_cfg import (
    PPOHECTORObservationsCfg, 
    HECTORBlindLocomotionObservationsCfg, 
    HECTORPerceptiveLocomotionObservationsCfg,
    HECTORGPUBlindLocomotionObservationsCfg, 
    HECTORRLBlindLocomotionObservationsCfg, 
)
from .observation_cfg import (
    TeacherObsCfg, 
    StudentObsCfg,
)
from .reward_cfg import (
    HECTORBlindLocomotionRewardsCfg, 
    HECTORPerceptiveLocomotionRewardsCfg, 
    HECTORSlipRewardsCfg, 
    HECTORGPUBlindLocomotionRewardsCfg, 
    HECTORRLBlindLocomotionRewardsCfg,
)
from .termination_cfg import HECTORTerminationsCfg
from .scene_cfg import (
    HECTORBlindLocomotionSceneCfg, 
    HECTORPerceptiveLocomotionSceneCfg, 
    HECTORSlipSceneCfg,
)