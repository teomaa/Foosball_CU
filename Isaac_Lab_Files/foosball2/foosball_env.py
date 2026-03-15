# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from numpy import random
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.foosball import FOOSBALL_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg,RigidObjectCfg,RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg,PhysxCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils.math import sample_uniform




@configclass
class FoosballEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    prismatic_action_scale=8
    revolute_action_scale=3
    action_space = 8
    observation_space = 41
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = FOOSBALL_CFG.replace(prim_path="/World/envs/env_.*/Foosball")


    # in-game Ball
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.SphereCfg(
            radius=0.024,  #0.01725,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1, 0.75, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.2,dynamic_friction=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                sleep_threshold=0.005,
                stabilization_threshold=0.01,
                max_depenetration_velocity=100.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=5.0),  #changed ball density
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.05, 0.0, 0.79025), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)


class FoosballEnv(DirectRLEnv):
    cfg: FoosballEnvCfg

    def __init__(self, cfg: FoosballEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        white_prismatic_joint_names = [
            "Keeper_W_PrismaticJoint",
            "Defense_W_PrismaticJoint",
            "Mid_W_PrismaticJoint",
            "Offense_W_PrismaticJoint",
        ]
        
        white_revolute_joint_names = [
            "Keeper_W_RevoluteJoint",
            "Defense_W_RevoluteJoint",
            "Mid_W_RevoluteJoint",
            "Offense_W_RevoluteJoint",
        ]


        black_prismatic_joint_names = [
            "Keeper_B_PrismaticJoint",
            "Defense_B_PrismaticJoint",
            "Mid_B_PrismaticJoint",
            "Offense_B_PrismaticJoint",
        ]
        
        black_revolute_joint_names = [
            "Keeper_B_RevoluteJoint",
            "Defense_B_RevoluteJoint",
            "Mid_B_RevoluteJoint",
            "Offense_B_RevoluteJoint",
        ]

        self.white_prismatic_dof_indices = list()
        for joint_name in white_prismatic_joint_names:
            self.white_prismatic_dof_indices.append(self.robot.joint_names.index(joint_name))
        self.white_prismatic_dof_indices.sort()

        self.white_revolute_dof_indices = list()
        for joint_name in white_revolute_joint_names:
            self.white_revolute_dof_indices.append(self.robot.joint_names.index(joint_name))
        self.white_revolute_dof_indices.sort()
    
        self.black_prismatic_dof_indices = list()
        for joint_name in black_prismatic_joint_names:
            self.black_prismatic_dof_indices.append(self.robot.joint_names.index(joint_name))
        self.black_prismatic_dof_indices.sort()

        self.black_revolute_dof_indices = list()
        for joint_name in black_revolute_joint_names:
            self.black_revolute_dof_indices.append(self.robot.joint_names.index(joint_name))
        self.black_revolute_dof_indices.sort()

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_velocities = self.object.data.root_vel_w

        self.prismatic_action_scale = self.cfg.prismatic_action_scale
        self.revolute_action_scale = self.cfg.revolute_action_scale


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        #Add Ball to Scene
        self.object = RigidObject(self.cfg.object_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add Table to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions.clone(), -1.0, 1.0)


    def _apply_action(self) -> None:
        n_wp = len(self.white_prismatic_dof_indices)
        n_wr = len(self.white_revolute_dof_indices)

        wp = self.actions[:, 0:n_wp]   # prismatic
        wr = self.actions[:, n_wp:n_wp+n_wr]  # revolute

        self.robot.set_joint_effort_target(
            wp*self.prismatic_action_scale, joint_ids=self.white_prismatic_dof_indices
        )

        # Hard revolute joint position limits: clamp to ±π
        rev_pos = self.joint_pos[:, self.white_revolute_dof_indices]
        rev_limit = math.pi
        restoring_stiffness = 50.0

        over_max = (rev_pos > rev_limit).float()
        under_min = (rev_pos < -rev_limit).float()
        in_bounds = 1.0 - over_max - under_min

        agent_torque = wr * self.revolute_action_scale
        agent_torque = agent_torque * in_bounds \
            + agent_torque.clamp(max=0.0) * over_max \
            + agent_torque.clamp(min=0.0) * under_min

        restoring_torque = -restoring_stiffness * over_max * (rev_pos - rev_limit) \
                         + -restoring_stiffness * under_min * (rev_pos + rev_limit)

        self.robot.set_joint_effort_target(
            agent_torque + restoring_torque, joint_ids=self.white_revolute_dof_indices
        )

        assert not torch.isnan(self.actions).any(), "NaN in actions"
        assert not torch.isinf(self.actions).any(), "Inf actions!"
    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos,
                self.joint_vel,
                self.object_pos,
                self.object_velocities,
            ),
            dim=-1,
        )

        # Clamp observations to prevent gradient explosion and NaN in policy network
        obs = torch.clamp(obs, -20.0, 20.0)
        # Replace any NaN/Inf from physics glitches with zeros
        obs = torch.nan_to_num(obs, nan=0.0, posinf=20.0, neginf=-20.0)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
         
        base_reward = compute_rewards(
            self.object_pos,)
        #action_penalty=0.01 * torch.sum(self.actions**2, dim=-1)
        total_reward=base_reward#-action_penalty

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_velocities = self.object.data.root_vel_w
        time_out = self.episode_length_buf >= self.max_episode_length #- 1
        #ball off table score/fall
        off_table_height=0.65
        fell_off_table = self.object_pos[:,2]<=off_table_height
        ball_pop_height = 0.8
        ball_too_high = self.object_pos[:,2]>=ball_pop_height

        
        #print(f"Time Out: {time_out}")
        #print(f"off table: {out_of_bounds}")


        out_of_bounds = white_goal(self.object_pos) | black_goal(self.object_pos) |fell_off_table |ball_too_high

        
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        
        object_default_state = self.object.data.default_root_state.clone()[env_ids]

        #noise for ball position
        #pos_noise = sample_uniform(-0.05, 0.05, (len(env_ids), 2), device=self.device)
        #vel_noise=sample_uniform(-1, 1, (len(env_ids), 2), device=self.device)

        object_default_state[:, :3] += self.scene.env_origins[env_ids]
        #object_default_state[:, :2] += pos_noise
        #object_default_state[:, 7:9] += vel_noise

        
        
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        
        
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        ##set black in up position
        #black_pos_up = 2*torch.ones(joint_pos.shape[0],len(self.black_revolute_dof_indices), device=self.device)
        #black_vel_up = torch.zeros(joint_pos.shape[0],len(self.black_revolute_dof_indices), device=self.device)
        #self.robot.write_joint_state_to_sim(black_pos_up, black_vel_up, self.black_revolute_dof_indices, env_ids)
   




@torch.jit.script
def white_goal(object_pos: torch.Tensor) -> torch.Tensor:
   
    return  object_pos[:, 0] < -0.61726


@torch.jit.script
def black_goal(object_pos: torch.Tensor) -> torch.Tensor:

    return object_pos[:, 0] > 0.61726 


@torch.jit.script
def ball_pop(object_pos: torch.Tensor) -> torch.Tensor:
    
    return object_pos[:, 2] >= 0.8

@torch.jit.script
def compute_rewards(
    object_pos: torch.Tensor,
    
    
):
    device = object_pos.device
    score = torch.zeros(object_pos.shape[0], dtype=torch.float32, device=device)
    
    # Check if white team scored a goal
    score[white_goal(object_pos)] = 100

    # Check if black team scored a goal
    score[black_goal(object_pos)] = -100

    score[ball_pop(object_pos)] = -50



    z = torch.zeros_like(object_pos[:, 1])
    y_dist = torch.pow(torch.max(torch.abs(object_pos[:, 1]) - 0.08525, z), 2)
    #print(f"Y dist: {y_dist}")
    x_dist_to_goal_white= torch.pow(object_pos[:, 0] + 0.61725, 2)
    #print(f"X dist: {x_dist_to_goal_white}")
    dist_to_goal_white= torch.sqrt(x_dist_to_goal_white + y_dist)

    
    #dist_goal_clamped=torch.clamp(dist_to_goal_white, min=1e-4, max=2.0)
    
    #Revised Reward
    #action_penalty = 0.01 * torch.sum(self.actions**2, dim=-1)
    dist_reward= 5.0 / (1.0 + 10.0 * dist_to_goal_white)
    total_reward=dist_reward
    
    
    #print(f"dist: {dist_reward}")
    #print(f"score: {score}")
    #print(f"reward: {total_reward}")
    #total_reward = score+1000*torch.exp(-5*dist_goal_clamped)
    assert not torch.isnan(total_reward).any(), "NaN in reward!"
    assert not torch.isinf(total_reward).any(), "Inf in reward!"

    return total_reward
