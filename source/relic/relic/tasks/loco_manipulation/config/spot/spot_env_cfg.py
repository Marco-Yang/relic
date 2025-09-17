# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from isaaclab.utils import configclass
from relic.tasks.loco_manipulation.interlimb_env_cfg import InterlimbEnvCfg

##
# Pre-defined configs
##
from relic.assets.spot.spot import SPOT_ARM_CFG  # isort: skip


@configclass
class SpotInterlimbEnvCfg_Phase_1(InterlimbEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to spot-arm
        self.scene.robot = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.joint_drive.gains.stiffness = None
        # self.rewards.foot_clearance = None


@configclass
class SpotInterlimbEnvCfg_Phase_2(InterlimbEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to spot-arm
        self.scene.robot = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # change event and termination terms
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        self.terminations.undesired_ground_contact.params["sensor_cfg"].body_names = [
            ".*uleg"
        ]

        # change the reward weights
        self.rewards.dof_torques_l2.weight = (-1.0e-4,)  # -1.0e-05
        self.rewards.dof_acc_l2.weight = (-1.0e-6,)  # -2.5e-7
        self.rewards.action_rate_l2.weight = -0.1  # -0.01
        self.rewards.track_base_orientation_l2.weight = -7.0
        self.rewards.track_base_height_l2.weight = -30.0
        self.rewards.flight_penalty.weight = -10.0
        self.rewards.foot_clearance.weight = 1.0

        self.rewards.foot_slip.weight = -1.0


@configclass
class SpotInterlimbEnvCfg_Phase_3(InterlimbEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to spot-arm
        self.scene.robot = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # change event and termination terms
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)
        self.terminations.undesired_ground_contact.params["sensor_cfg"].body_names = [
            ".*uleg"
        ]

        # change the reward weights
        # self.rewards.dof_torques_l2.weight = -5.0e-4 # -1.0e-4, -1.0e-05
        # self.rewards.dof_acc_l2.weight = -2.0e-6 # -1.0e-6, -2.5e-7
        self.rewards.dof_torques_l2.params["leg_ctl_weight"] = -2.0e-4
        self.rewards.dof_acc_l2.params["leg_ctl_weight"] = -1.0e-6

        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.track_base_orientation_l2.weight = -45.0
        self.rewards.track_base_height_l2.weight = -120.0
        self.rewards.flight_penalty.weight = -10.0
        self.rewards.foot_clearance.weight = 1.0
        self.rewards.foot_slip.weight = -2.0

        self.rewards.feet_air_time_target.weight = 0.5
        self.rewards.air_time_variance.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = -0.0
        self.rewards.ang_vel_xy_l2.weight = -0.0


@configclass
class SpotInterlimbEnvCfg_PLAY(SpotInterlimbEnvCfg_Phase_1):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 128
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.curriculum.terrain_levels = None

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.0)
        self.commands.base_velocity.debug_vis = False

        self.terminations.undesired_ground_contact.params["sensor_cfg"].body_names = [
            ".*uleg"
        ]
        self.terminations.base_contact = None
