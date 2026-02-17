import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
import glfw

from ai_agents.v2.gym.mujoco_table_render_mixin import MujocoTableRenderMixin
from ai_agents.v2.gym.strategies import make_strategy

DIRECTION_CHANGE = 1
TABLE_MAX_Y_DIM = 65
BALL_STOPPED_COUNT_THRESHOLD = 10
MAX_STEPS = 40
SIM_PATH = os.environ.get('SIM_PATH', './foosball_sim/v2/foosball_sim.xml')

RODS = ["_goal_", "_def_", "_mid_", "_attack_"]

class FoosballEnv( MujocoTableRenderMixin, gym.Env, ):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, antagonist_model=None, play_until_goal=False, verbose_mode=False,
                 protagonist_strategy_name="basic", antagonist_strategy_name="basic"):
        super(FoosballEnv, self).__init__()

        dir_path = os.path.dirname(os.path.realpath(__file__))
        xml_file = SIM_PATH

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        self.simulation_time = 0

        self.num_rods_per_player = 4
        self.num_players = 2
        self.num_rods = self.num_rods_per_player * self.num_players  # Total rods

        self.protagonist_action_size = self.num_rods_per_player * 2  # 8 actions for protagonist
        self.antagonist_action_size = self.num_rods_per_player * 2   # 8 actions for antagonist

        action_high = np.ones(self.protagonist_action_size)
        self.rotation_action_space = spaces.Box(
            low=-2.5 * action_high, high=2.5 * action_high, dtype=np.float32
        )

        self.goal_linear_action_space = spaces.Box(
            low=-10.0 * action_high, high=10.0 * action_high, dtype=np.float32
        )
        self.def_linear_action_space = spaces.Box(
            low=-20.0 * action_high, high=20.0 * action_high, dtype=np.float32
        )
        self.mid_linear_action_space = spaces.Box(
            low=-7.0 * action_high, high=7.0 * action_high, dtype=np.float32
        )
        self.attack_linear_action_space = spaces.Box(
            low=-12.0 * action_high, high=12.0 * action_high, dtype=np.float32
        )

        # TEMP
        self.action_space = spaces.Box(
            low=-20 * action_high, high=20 * action_high, dtype=np.float32
        )

        obs_dim = 38
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.viewer = None

        self._healthy_reward = 1.0
        self._ctrl_cost_weight = 0.005
        self._terminate_when_unhealthy = True
        self._healthy_z_range = (-80, 80)
        self.max_no_progress_steps = 15

        self.prev_ball_y = None
        self.no_progress_steps = 0
        self.ball_stopped_count = 0

        self.antagonist_model = antagonist_model
        self.play_until_goal = play_until_goal
        self.verbose_mode = verbose_mode

        self.slide_gain = 5.0  # >1 => move more each step; try 3–10
        self.slide_step_clip = 7.0  # max |delta slide| per step (joint units)

        # --- Caching for performance optimization ---
        self._initialize_caches()

        # --- Pluggable strategies ---
        self.protagonist_strategy = make_strategy(protagonist_strategy_name, self, "y")
        self.antagonist_strategy = make_strategy(antagonist_strategy_name, self, "b")

    def _initialize_caches(self):
        """Initialize all caches for expensive lookups to avoid repeated mj_name2id calls."""
        # Cache ball joint IDs and addresses
        ball_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_x')
        ball_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_y')
        ball_z_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_z')
        
        self._cache_ball_x_id = ball_x_id
        self._cache_ball_y_id = ball_y_id
        self._cache_ball_z_id = ball_z_id
        self._cache_ball_x_qpos_adr = self.model.jnt_qposadr[ball_x_id]
        self._cache_ball_y_qpos_adr = self.model.jnt_qposadr[ball_y_id]
        self._cache_ball_z_qpos_adr = self.model.jnt_qposadr[ball_z_id]
        self._cache_ball_x_qvel_adr = self.model.jnt_dofadr[ball_x_id]
        self._cache_ball_y_qvel_adr = self.model.jnt_dofadr[ball_y_id]
        self._cache_ball_z_qvel_adr = self.model.jnt_dofadr[ball_z_id]

        # Cache body IDs for all rods (protagonist and antagonist)
        # Note: rod_key in action methods is "goal", "def", "mid", "attack" (no underscores)
        # but body names use "_goal_", "_def_", etc. We need to handle both formats
        self._cache_guy_body_ids = {}
        rod_keys_clean = ["goal", "def", "mid", "attack"]  # Clean keys used in action methods
        for team in ['y', 'b']:
            for rod_key_clean in rod_keys_clean:
                key = f"{team}{rod_key_clean}"
                rod_key_with_underscores = f"_{rod_key_clean}_"  # For body name lookup
                bids = []
                for i in range(1, 8):
                    # Body names use format: "y_goal_guy1" (with underscores)
                    nm = f"{team}{rod_key_with_underscores}guy{i}"
                    try:
                        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nm)
                        if bid >= 0:
                            bids.append(bid)
                    except Exception:
                        pass
                self._cache_guy_body_ids[key] = bids

        # Cache actuator and joint information for all rods
        self._cache_rod_info = {}
        for team in ['y', 'b']:
            for rod_key_clean in rod_keys_clean:
                key = f"{team}{rod_key_clean}"
                # Actuator names use underscores: "y_goal_linear", not "y_goal_linear"
                act_lin_name = f"{team}_{rod_key_clean}_linear"
                act_rot_name = f"{team}_{rod_key_clean}_rotation"
                
                act_lin_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_lin_name)
                act_rot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_rot_name)
                
                if act_lin_id >= 0 and act_rot_id >= 0:
                    jnt_id_lin = int(self.model.actuator_trnid[act_lin_id, 0])
                    jnt_id_rot = int(self.model.actuator_trnid[act_rot_id, 0])
                    
                    self._cache_rod_info[key] = {
                        'act_lin_id': act_lin_id,
                        'act_rot_id': act_rot_id,
                        'jnt_id_lin': jnt_id_lin,
                        'jnt_id_rot': jnt_id_rot,
                        'qpos_adr_lin': int(self.model.jnt_qposadr[jnt_id_lin]),
                        'qpos_adr_rot': int(self.model.jnt_qposadr[jnt_id_rot]),
                        'lin_min': float(self.model.actuator_ctrlrange[act_lin_id, 0]),
                        'lin_max': float(self.model.actuator_ctrlrange[act_lin_id, 1]),
                        'rot_min': float(self.model.actuator_ctrlrange[act_rot_id, 0]),
                        'rot_max': float(self.model.actuator_ctrlrange[act_rot_id, 1]),
                    }
                else:
                    self._cache_rod_info[key] = None

        # Pre-compute finite difference approximations for slide sensitivity
        # Instead of computing dx/ds every time, use a cached approximation
        # For rods, dx/ds is approximately constant (rod moves foosmen in X direction)
        # We'll compute it once per rod and cache it
        self._cache_dx_ds = {}
        mujoco.mj_forward(self.model, self.data)  # Ensure physics is up-to-date for initial computation
        
        for team in ['y', 'b']:
            for rod_key_clean in rod_keys_clean:
                key = f"{team}{rod_key_clean}"
                if key in self._cache_rod_info and self._cache_rod_info[key] is not None:
                    rod_info = self._cache_rod_info[key]
                    guy_bids = self._cache_guy_body_ids.get(key, [])
                    if guy_bids:
                        # Compute dx/ds once and cache it (it's approximately constant for linear rods)
                        qpos_adr = rod_info['qpos_adr_lin']
                        # Use middle foosman as reference
                        idx_ref = len(guy_bids) // 2
                        dx_ds = self._compute_dx_ds_once(qpos_adr, guy_bids, idx_ref)
                        self._cache_dx_ds[key] = dx_ds
                    else:
                        self._cache_dx_ds[key] = 1.0  # Default fallback
                else:
                    self._cache_dx_ds[key] = 1.0  # Default fallback
        

    def _compute_dx_ds_once(self, qpos_adr: int, guy_bids: list, idx_ref: int) -> float:
        """Compute dx/ds once for caching. This is expensive but only done once at init."""
        qpos_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()
        try:
            s0 = float(self.data.qpos[qpos_adr])
            x0 = float(self.data.xpos[guy_bids[idx_ref], 0])

            eps = 1e-3
            self.data.qpos[qpos_adr] = s0 + eps
            mujoco.mj_forward(self.model, self.data)
            x1 = float(self.data.xpos[guy_bids[idx_ref], 0])

            return (x1 - x0) / eps
        finally:
            self.data.qpos[:] = qpos_backup
            self.data.qvel[:] = qvel_backup
            mujoco.mj_forward(self.model, self.data)

    def set_antagonist_model(self, antagonist_model):
        self.antagonist_model = antagonist_model

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        ball_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_x')
        ball_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ball_y')

        x_qpos_adr = self.model.jnt_qposadr[ball_x_id]
        y_qpos_adr = self.model.jnt_qposadr[ball_y_id]

        xy_random = np.random.normal(
            loc=[-0.5, 0.0],
            scale=[0.5, 0.5]
        )

        self.data.qpos[x_qpos_adr] = xy_random[0]
        self.data.qpos[y_qpos_adr] = xy_random[1]

        self.simulation_time = 0
        self.prev_ball_y = self.data.qpos[y_qpos_adr]
        self.no_progress_steps = 0
        self.ball_stopped_count = 0

        self.simulation_time = 0
        self.prev_ball_y = self.data.qpos[y_qpos_adr]
        self.no_progress_steps = 0
        self.ball_stopped_count = 0

        # Reset strategy state each episode
        self.protagonist_strategy.reset()
        self.antagonist_strategy.reset()

        return self._get_obs(), {}

    def step(self, protagonist_action):
        protagonist_action = np.clip(protagonist_action, self.action_space.low, self.action_space.high)

        antagonist_observation = self._get_antagonist_obs()

        if self.antagonist_model is not None:
            antagonist_action = self.antagonist_model.predict(antagonist_observation)
            antagonist_action = np.clip(antagonist_action, -1.0, 1.0)

            antagonist_action = self._adjust_antagonist_action(antagonist_action)
        else:
            antagonist_action = np.zeros(self.antagonist_action_size)

        self.data.ctrl[:self.protagonist_action_size] = protagonist_action
        self.data.ctrl[self.protagonist_action_size:self.protagonist_action_size + self.antagonist_action_size] = antagonist_action

        mujoco.mj_step(self.model, self.data)
        self.simulation_time += self.model.opt.timestep
        
        # Call mj_forward ONCE after step to update positions/velocities for observations
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()

        reward = self.compute_reward(protagonist_action)
        terminated = self.terminated

        info = {}

        return obs, reward, terminated, False, info

    def _get_ball_obs(self):
        # Use cached addresses instead of repeated lookups
        ball_pos = [
            self.data.qpos[self._cache_ball_x_qpos_adr],
            self.data.qpos[self._cache_ball_y_qpos_adr],
            self.data.qpos[self._cache_ball_z_qpos_adr]
        ]
        ball_vel = [
            self.data.qvel[self._cache_ball_x_qvel_adr],
            self.data.qvel[self._cache_ball_y_qvel_adr],
            self.data.qvel[self._cache_ball_z_qvel_adr]
        ]

        return ball_pos, ball_vel

    def _get_antagonist_obs(self):
        return self._get_obs()

    def _get_obs(self):
        ball_pos, ball_vel = self._get_ball_obs()

        rod_slide_positions = []
        rod_slide_velocities = []
        rod_rotate_positions = []
        rod_rotate_velocities = []

        # Collect observations for both players' rods
        for player in ['y', 'b']:
            for rod in RODS:
                # Linear joints
                slide_joint_name = f"{player}{rod}linear"
                slide_joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, slide_joint_name
                )
                slide_qpos_adr = self.model.jnt_qposadr[slide_joint_id]
                slide_qvel_adr = self.model.jnt_dofadr[slide_joint_id]
                rod_slide_positions.append(self.data.qpos[slide_qpos_adr])
                rod_slide_velocities.append(self.data.qvel[slide_qvel_adr])

                # Rotational joints
                rotate_joint_name = f"{player}{rod}rotation"
                rotate_joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, rotate_joint_name
                )
                rotate_qpos_adr = self.model.jnt_qposadr[rotate_joint_id]
                rotate_qvel_adr = self.model.jnt_dofadr[rotate_joint_id]
                rod_rotate_positions.append(self.data.qpos[rotate_qpos_adr])
                rod_rotate_velocities.append(self.data.qvel[rotate_qvel_adr])

        obs = np.concatenate([
            ball_pos,
            ball_vel,
            rod_slide_positions,
            rod_slide_velocities,
            rod_rotate_positions,
            rod_rotate_velocities
        ])

        assert obs.shape == self.observation_space.shape, (
            f"Observation shape {obs.shape} does not match observation space shape {self.observation_space.shape}"
        )

        return obs

    def _adjust_antagonist_action(self, antagonist_action):
        # adjusted_action = -antagonist_action.copy()
        # return adjusted_action
        return antagonist_action


    def euclidean_goal_distance(self, x, y):
        # Point (0, 64)
        return math.sqrt((x - 0) ** 2 + (y - TABLE_MAX_Y_DIM) ** 2)

    def compute_reward(self, protagonist_action):
        ball_obs = self._get_ball_obs()
        ball_y = ball_obs[0][1]
        ball_x = ball_obs[0][0]

        inverse_distance_to_goal = 300 - self.euclidean_goal_distance(ball_x, ball_y)

        if ball_y >  TABLE_MAX_Y_DIM:
            inverse_distance_to_goal = 0

        ctrl_cost = self.control_cost(protagonist_action)

        victory = 1000 * DIRECTION_CHANGE if ball_y >  TABLE_MAX_Y_DIM else 0  # Ball in antagonist's goal
        loss = -1000 * DIRECTION_CHANGE if ball_y < -1.0 * TABLE_MAX_Y_DIM else 0  # Ball in protagonist's goal

        reward = loss + victory + inverse_distance_to_goal

        return reward

    @property
    def healthy_reward(self):
        return (
                float(self.is_healthy or self._terminate_when_unhealthy)
                * self._healthy_reward
        )

    def control_cost(self, action):
        # 2-norm
        #control_cost = self._ctrl_cost_weight * np.sum(np.square(action)) * -1.0

        # 1-norm
        control_cost = self._ctrl_cost_weight * np.sum(np.abs(action)) * -1.0

        # L0 norm
        #control_cost = self._ctrl_cost_weight * np.count_nonzero(action) * -1.0

        return control_cost

    @property
    def is_healthy(self):
        ball_z = self._get_ball_obs()[0][2]

        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < ball_z < max_z

        return is_healthy

    def _is_ball_moving(self):
        ball_pos, ball_vel = self._get_ball_obs()

        return np.linalg.norm(ball_vel) > 0.5

    def _determine_progression(self):
        ball_y = self._get_ball_obs()[0][1]

        if self.prev_ball_y is not None:
            if ball_y > self.prev_ball_y:
                self.no_progress_steps = 0
            else:
                self.no_progress_steps += 1

        self.prev_ball_y = ball_y

    @property
    def terminated(self):
        self._determine_progression()

        self.ball_stopped_count = 0 if self._is_ball_moving() else self.ball_stopped_count + 1
        ball_stagnant = self.ball_stopped_count >= BALL_STOPPED_COUNT_THRESHOLD

        over_max_steps = self.simulation_time >= MAX_STEPS

        unhealthy = not self.is_healthy
        no_progress = self.no_progress_steps >= self.max_no_progress_steps

        ball_y = self._get_ball_obs()[0][1]
        ball_x = self._get_ball_obs()[0][0]

        victory = ball_y < -1 * TABLE_MAX_Y_DIM or ball_y > TABLE_MAX_Y_DIM  # Ball in any goal

        if victory:
            print("Victory")
            print(f"Ball x: {ball_x}, Ball y: {ball_y}")

        terminated = (
                unhealthy or (no_progress and not self.play_until_goal) or ball_stagnant or over_max_steps #or victory
        ) if self._terminate_when_unhealthy else False

        if self.verbose_mode and terminated:
            print("Terminated")
            print(f"Unhealthy: {unhealthy}, No progress: {no_progress}, Victory: {victory}, Ball stagnant: {ball_stagnant}")
            print("x: ", ball_x, "y: ", ball_y)
        return terminated

    def protagonist_all_rods_action_toward_ball(self, obs: np.ndarray) -> np.ndarray:
        """Build an 8-d action for protagonist (yellow). Delegates to strategy."""
        return self.protagonist_strategy.compute_action(obs)

    def antagonist_all_rods_action_toward_ball(self, obs: np.ndarray) -> np.ndarray:
        """Build an 8-d action for antagonist (black). Delegates to strategy."""
        return self.antagonist_strategy.compute_action(obs)

    # ------------------------ helpers (private) ------------------------

    def _slide_target_for_rod_toward_ball(self, team: str, rod_key: str, actuator_name: str,
                                          ball_x: float, ball_y: float) -> float:
        """
        Desired SLIDE setpoint so the rod moves aggressively toward aligning its closest
        foosman's world-X with ball_x. Uses cached values and cached dx/ds approximation.
        """
        key = f"{team}{rod_key}"
        rod_info = self._cache_rod_info.get(key)
        if rod_info is None:
            return 0.0

        guy_bids = self._rod_guy_body_ids(team, rod_key)
        if not guy_bids:
            return 0.0

        # Physics state is already up-to-date from step() - no need to call mj_forward here
        men_xy = np.array([[self.data.xpos[bid, 0], self.data.xpos[bid, 1]] for bid in guy_bids],
                          dtype=np.float64)
        d2 = (men_xy[:, 0] - ball_x) ** 2 + (men_xy[:, 1] - ball_y) ** 2
        idx_closest = int(np.argmin(d2))
        guy_x = float(men_xy[idx_closest, 0])

        s0 = float(self.data.qpos[rod_info['qpos_adr_lin']])
        # Use cached dx/ds instead of expensive finite difference computation
        dx_ds = self._cache_dx_ds.get(key, 1.0)

        if abs(dx_ds) < 1e-6:
            direction = np.sign(ball_x - guy_x) if ball_x != guy_x else 0.0
            ideal_delta_s = direction
        else:
            ideal_delta_s = (ball_x - guy_x) / dx_ds

        delta_s = self.slide_gain * ideal_delta_s
        if self.slide_step_clip is not None:
            delta_s = float(np.clip(delta_s, -self.slide_step_clip, self.slide_step_clip))

        s_target = s0 + delta_s
        return float(np.clip(s_target, rod_info['lin_min'], rod_info['lin_max']))

    def _nearest_guy_stats_and_rot_limits(self, team: str, rod_key: str,
                                          actuator_lin_name: str,
                                          actuator_rot_name: str):
        """
        Returns:
          (guy_x, guy_y,
           qpos_adr_rot, rot_min, rot_max,
           qpos_adr_lin, lin_min, lin_max)
        for the nearest foosman on this rod, or None if something is missing.
        Uses cached values for all lookups.
        """
        key = f"{team}{rod_key}"
        rod_info = self._cache_rod_info.get(key)
        if rod_info is None:
            return None

        guy_bids = self._rod_guy_body_ids(team, rod_key)
        if not guy_bids:
            return None

        # Use cached ball addresses
        bx = float(self.data.qpos[self._cache_ball_x_qpos_adr])
        by = float(self.data.qpos[self._cache_ball_y_qpos_adr])

        # Physics state is already up-to-date from step() - no need to call mj_forward here
        men_xy = np.array([[self.data.xpos[bid, 0], self.data.xpos[bid, 1]] for bid in guy_bids],
                          dtype=np.float64)
        d2 = (men_xy[:, 0] - bx) ** 2 + (men_xy[:, 1] - by) ** 2
        idx_closest = int(np.argmin(d2))

        guy_x, guy_y = float(men_xy[idx_closest, 0]), float(men_xy[idx_closest, 1])

        return (guy_x, guy_y,
                rod_info['qpos_adr_rot'], rod_info['rot_min'], rod_info['rot_max'],
                rod_info['qpos_adr_lin'], rod_info['lin_min'], rod_info['lin_max'])

    def _rod_guy_body_ids(self, team: str, rod_key: str) -> list:
        """
        Resolve body ids for foosmen on a given rod. Uses cached values.
        """
        key = f"{team}{rod_key}"
        return self._cache_guy_body_ids.get(key, [])

    def _finite_diff_guy_x_wrt_slide(self, qpos_adr: int, guy_bids: list, idx_closest: int) -> float:
        """
        Compute dx/ds for the chosen guy via finite difference on the rod's slide qpos.
        NOTE: This method is now deprecated in favor of cached dx/ds values.
        Kept for backward compatibility but should not be called in optimized code paths.
        """
        qpos_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()
        try:
            s0 = float(self.data.qpos[qpos_adr])
            # Physics state is already up-to-date from step() - use current xpos directly
            x0 = float(self.data.xpos[guy_bids[idx_closest], 0])

            eps = 1e-3
            self.data.qpos[qpos_adr] = s0 + eps
            mujoco.mj_forward(self.model, self.data)
            x1 = float(self.data.xpos[guy_bids[idx_closest], 0])

            return (x1 - x0) / eps
        finally:
            self.data.qpos[:] = qpos_backup
            self.data.qvel[:] = qvel_backup
            mujoco.mj_forward(self.model, self.data)
