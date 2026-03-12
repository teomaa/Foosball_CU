# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a Foosball Table."""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

FOOSBALL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=r"C:\Users\richa\Downloads\Foosball_no_ball.usd",  #current USD Location
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            #max_linear_velocity=1000.0,
            #max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "Keeper_W_PrismaticJoint": 0.0,
            "Defense_W_PrismaticJoint": 0.0,
            "Mid_W_PrismaticJoint": 0.0,
            "Offense_W_PrismaticJoint": 0.0,
            "Keeper_B_PrismaticJoint": 0.0,
            "Defense_B_PrismaticJoint": 0.0,
            "Mid_B_PrismaticJoint": 0.0,
            "Offense_B_PrismaticJoint": 0.0,
            "Keeper_W_RevoluteJoint": 0.0,
            "Defense_W_RevoluteJoint": 0.0,
            "Mid_W_RevoluteJoint": 0.0,
            "Offense_W_RevoluteJoint": 0.0,
            "Keeper_B_RevoluteJoint": 0.0,
            "Defense_B_RevoluteJoint": 0.0,
            "Mid_B_RevoluteJoint": 0.0,
            "Offense_B_RevoluteJoint": 0.0,

        },
    ),
        actuators={
        "white_joints": ImplicitActuatorCfg(
            joint_names_expr=["Keeper_W_PrismaticJoint","Defense_W_PrismaticJoint","Mid_W_PrismaticJoint","Offense_W_PrismaticJoint","Keeper_W_RevoluteJoint","Defense_W_RevoluteJoint","Mid_W_RevoluteJoint","Offense_W_RevoluteJoint"],
            velocity_limit=20.0,
            effort_limit=40,
            stiffness=0.0,
            damping=10.0,
        ),
        "black_joints": ImplicitActuatorCfg(
           joint_names_expr=["Keeper_B_PrismaticJoint","Defense_B_PrismaticJoint","Mid_B_PrismaticJoint","Offense_B_PrismaticJoint","Keeper_B_RevoluteJoint","Defense_B_RevoluteJoint","Mid_B_RevoluteJoint","Offense_B_RevoluteJoint"],
            velocity_limit=20.0,
            effort_limit=40,
            stiffness=0.0,
            damping=10.0,
        ),
    },

)
