# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from numpy import random
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.foosball import FOOSBALL_CFG, FOOSBALL_VS_CFG

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
    episode_length_s = 10.0 #5 12/19
    prismatic_action_scale=  8#12#10#160
    revolute_action_scale= 8#3#5#12#4#40
    action_space = 8
    observation_space = 41
    state_space = 0
    revolute_action_penalty: bool = False  # Penalize revolute actions (0.05 * sum of squares)

    # opponent (frozen model)
    opponent_checkpoint: str | None = None    # Path to SB3 PPO .zip
    opponent_deterministic: bool = True       # Deterministic opponent actions

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
            radius=0.0175,  #0.01725,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1, 0.75, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.4,dynamic_friction=0.25),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.005,
                stabilization_threshold=0.01,
                max_depenetration_velocity=2.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=380.0),  #changed ball density
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.05, 0.0, 0.79025), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.0, replicate_physics=True)


@configclass
class FoosballVsEnvCfg(FoosballEnvCfg):
    robot_cfg: ArticulationCfg = FOOSBALL_VS_CFG.replace(prim_path="/World/envs/env_.*/Foosball")


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
        #Add goal tracker to tensorboard
        self.goal_scored = torch.zeros(self.scene.num_envs, device=self.device, dtype=torch.float32)
        self.black_goal_scored = torch.zeros(self.scene.num_envs, device=self.device, dtype=torch.float32)

        # --- Frozen opponent model ---
        self.opponent_model = None
        if self.cfg.opponent_checkpoint is not None:
            from stable_baselines3 import PPO
            self.opponent_model = PPO.load(self.cfg.opponent_checkpoint, device=self.device)
            self.opponent_model.policy.set_training_mode(False)
            print(f"[INFO] Loaded frozen opponent from {self.cfg.opponent_checkpoint}")

            # Build mirror index/sign tensors for observation mirroring.
            # The frozen model was trained as white; to play as black we swap
            # white<->black joint indices and negate ball_pos_x / ball_vel_x.
            #
            # Obs layout: [joint_pos(16), joint_vel(16), ball_pos(3), ball_vel(6)] = 41
            # joint order (sorted DOF indices): 8 white then 8 black (or interleaved —
            # we build the mapping from actual DOF indices).

            n_joints = 16  # total joints
            # Build white->black and black->white index mapping for joints
            wp = self.white_prismatic_dof_indices  # 4 indices
            wr = self.white_revolute_dof_indices    # 4 indices
            bp = self.black_prismatic_dof_indices   # 4 indices
            br = self.black_revolute_dof_indices    # 4 indices

            joint_swap = list(range(n_joints))
            # Swap each white joint with corresponding black joint (same rod type)
            for w, b in zip(wp, bp):
                joint_swap[w] = b
                joint_swap[b] = w
            for w, b in zip(wr, br):
                joint_swap[w] = b
                joint_swap[b] = w

            mirror_indices = list(range(41))
            # joint_pos block (0..15): swap white<->black
            for i in range(n_joints):
                mirror_indices[i] = joint_swap[i]
            # joint_vel block (16..31): swap white<->black
            for i in range(n_joints):
                mirror_indices[16 + i] = 16 + joint_swap[i]
            # ball_pos (32,33,34) and ball_vel (35..40) stay in place

            self.mirror_indices = torch.tensor(mirror_indices, device=self.device, dtype=torch.long)

            mirror_signs = torch.ones(41, device=self.device, dtype=torch.float32)
            mirror_signs[32] = -1.0  # ball_pos_x
            mirror_signs[35] = -1.0  # ball_vel_x
            self.mirror_signs = mirror_signs


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
        # Strong restoring torque + zero out agent effort when past limit
        rev_pos = self.joint_pos[:, self.white_revolute_dof_indices]
        rev_limit = math.pi
        restoring_stiffness = 50.0

        over_max = (rev_pos > rev_limit).float()
        under_min = (rev_pos < -rev_limit).float()
        in_bounds = 1.0 - over_max - under_min

        # Only allow agent torque that pushes back toward center when at limits
        agent_torque = wr * self.revolute_action_scale
        # Zero out torque pushing further past limits
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

        # --- Frozen opponent actions (black team) ---
        if self.opponent_model is not None:
            raw_obs = torch.cat((self.joint_pos, self.joint_vel, self.object_pos, self.object_velocities), dim=-1)
            raw_obs = torch.clamp(raw_obs, -20.0, 20.0)
            raw_obs = torch.nan_to_num(raw_obs, nan=0.0, posinf=20.0, neginf=-20.0)
            mirrored_obs = self._mirror_obs_for_opponent(raw_obs)

            with torch.no_grad():
                opp_actions, _, _ = self.opponent_model.policy.forward(mirrored_obs, deterministic=self.cfg.opponent_deterministic)
            opp_actions = torch.clamp(opp_actions, -1.0, 1.0)

            opp_bp = opp_actions[:, 0:4]
            opp_br = opp_actions[:, 4:8]

            self.robot.set_joint_effort_target(
                opp_bp * self.prismatic_action_scale, joint_ids=self.black_prismatic_dof_indices
            )

            # Same revolute limit enforcement as white
            black_rev_pos = self.joint_pos[:, self.black_revolute_dof_indices]
            rev_limit = math.pi
            restoring_stiffness = 50.0

            over_max_b = (black_rev_pos > rev_limit).float()
            under_min_b = (black_rev_pos < -rev_limit).float()
            in_bounds_b = 1.0 - over_max_b - under_min_b

            opp_torque = opp_br * self.revolute_action_scale
            opp_torque = opp_torque * in_bounds_b \
                + opp_torque.clamp(max=0.0) * over_max_b \
                + opp_torque.clamp(min=0.0) * under_min_b

            restoring_torque_b = -restoring_stiffness * over_max_b * (black_rev_pos - rev_limit) \
                               + -restoring_stiffness * under_min_b * (black_rev_pos + rev_limit)

            self.robot.set_joint_effort_target(
                opp_torque + restoring_torque_b, joint_ids=self.black_revolute_dof_indices
            )
    def _mirror_obs_for_opponent(self, obs: torch.Tensor) -> torch.Tensor:
        """Mirror observations so a white-trained model can play as black."""
        return obs[:, self.mirror_indices] * self.mirror_signs

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
        total_reward = base_reward

        if self.cfg.revolute_action_penalty:
            revolute_actions = self.actions[:, 4:8]
            total_reward = total_reward - 0.05 * torch.sum(revolute_actions**2, dim=-1)

        # Penalize when opponent scores (ball in white's goal)
        if self.opponent_model is not None:
            opponent_scored = black_goal(self.object_pos)
            total_reward[opponent_scored] -= 10.0

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_velocities = self.object.data.root_vel_w
        time_out = self.episode_length_buf >= self.max_episode_length #- 1
        #ball off table score/fall
        off_table_height=0.5
        fell_off_table = self.object_pos[:,2]<=off_table_height
        ball_pop_height = 1
        ball_too_high = self.object_pos[:,2]>=ball_pop_height

        
        #print(f"Time Out: {time_out}")
        #print(f"off table: {out_of_bounds}")

        white_scored = white_goal(self.object_pos)
        black_scored = black_goal(self.object_pos)

        out_of_bounds = white_scored | black_scored | fell_off_table | ball_too_high
        self.goal_scored = white_scored.float()
        self.black_goal_scored = black_scored.float()

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

            # ----- LOGGING -----
        if self.extras is not None:
            self.extras["goal_scored_pct"] = torch.mean(self.goal_scored[env_ids])
            if self.opponent_model is not None:
                self.extras["opponent_goal_scored_pct"] = torch.mean(self.black_goal_scored[env_ids])

        # Reset tracker
        self.goal_scored[env_ids] = 0.0
        self.black_goal_scored[env_ids] = 0.0
        
        
        super()._reset_idx(env_ids)
        
        object_default_state = self.object.data.default_root_state.clone()[env_ids]

        #noise for ball position
        pos_noise = sample_uniform(-0.2, 0.2, (len(env_ids), 2), device=self.device)
        vel_noise=sample_uniform(-2, 2, (len(env_ids), 2), device=self.device)

        object_default_state[:, :3] += self.scene.env_origins[env_ids]
        object_default_state[:, :2] += pos_noise
        object_default_state[:, 7:9] += vel_noise

        
        
        object_default_state[:, 9:] = 0 #object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
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
   
    return  object_pos[:, 0] < -0.62 #-0.61726


@torch.jit.script
def black_goal(object_pos: torch.Tensor) -> torch.Tensor:

    return object_pos[:, 0] > 0.62 #0.61726 


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
    score[white_goal(object_pos)] = 10

    # Check if black team scored a goal
    #score[black_goal(object_pos)] = -100

    #score[ball_pop(object_pos)] = -50



    z = torch.zeros_like(object_pos[:, 1])
    y_dist = torch.pow(torch.max(torch.abs(object_pos[:, 1]) - 0.08525, z), 2)
    #print(f"Y dist: {y_dist}")
    x_dist_to_goal_white= torch.pow(object_pos[:, 0] + 0.61725, 2)
    #print(f"X dist: {x_dist_to_goal_white}")
    dist_to_goal_white= torch.sqrt(x_dist_to_goal_white + y_dist)

    dist_penalty = -1.0 * torch.tanh(3.0 * dist_to_goal_white)
    #dist_goal_clamped=torch.clamp(dist_to_goal_white, min=1e-4, max=2.0)
    
    #Revised Reward
    #action_penalty = 0.01 * torch.sum(self.actions**2, dim=-1)
    #dist_reward= 5.0 / (1.0 + 10.0 * dist_to_goal_white)
    
    #dist_reward = torch.clamp(dist_reward, max=4.0) 12_19 shaping attempt
    #dist_reward *= torch.exp(-3.0 * (0.62 - object_pos[:,0]).clamp(min=0)) #12_19 shaping attempt
    total_reward=dist_penalty +score
    
    
    #print(f"dist: {dist_reward}")
    #print(f"score: {score}")
    #print(f"reward: {total_reward}")
    #total_reward = score+1000*torch.exp(-5*dist_goal_clamped)
    assert not torch.isnan(total_reward).any(), "NaN in reward!"
    assert not torch.isinf(total_reward).any(), "Inf in reward!"

    return total_reward
