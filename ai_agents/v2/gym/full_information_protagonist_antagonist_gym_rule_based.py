import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
import glfw

from ai_agents.v2.gym.mujoco_table_render_mixin import MujocoTableRenderMixin

DIRECTION_CHANGE = 1
TABLE_MAX_Y_DIM = 65
BALL_STOPPED_COUNT_THRESHOLD = 10
MAX_STEPS = 40
SIM_PATH = os.environ.get('SIM_PATH', './foosball_sim/v2/foosball_sim.xml')

RODS = ["_goal_", "_def_", "_mid_", "_attack_"]

class FoosballEnv( MujocoTableRenderMixin, gym.Env, ):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, antagonist_model=None, play_until_goal=False, verbose_mode=False):
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

        # --- Kick params (editable) ---
        # Distance thresholds (table units) for X proximity
        self.kick_x_close = 2.0        # start cocking when |ball_x - guy_x| <= this
        self.kick_x_contact = 0.3      # start forward kick when |ball_x - guy_x| <= this

        # Y “in front” windows (ball should be slightly ahead of foosman toward +Y)
        self.kick_y_front_min = -1.2    # begin cocking when (ball_y - guy_y) in [min, max]
        self.kick_y_front_max = 9.0
        self.kick_y_contact_max = 1.25 # forward kick when 0 <= (ball_y - guy_y) <= this

        # Rotation magnitudes (radians, absolute targets)
        self.kick_back_angle = 1.95    # small back-cock angle
        self.kick_forward_angle = 3.90 # forward strike angle (≈2× back)

        # Sign of "back" vs "forward" (flip if your mesh is reversed)
        # With default +Y attack direction, positive usually means one direction around Z.
        # If they rotate the wrong way, set this to -1.0.
        self.kick_sign = 1.0

        self.slide_gain = 5.0  # >1 => move more each step; try 3–10
        self.slide_step_clip = 7.0  # max |delta slide| per step (joint units)

        # --- Angle-shooting offset (editable) ---
        # When kicking forward, override normal slide logic and offset the rod's X
        # by a random amount in [-kick_offset_max, kick_offset_max].
        self.kick_offset_max = 5.0

        # --- Lane clearing angle (editable) ---
        # When a back rod shoots, rods in front rotate "up" by this angle so the ball can pass.
        self.clear_lane_angle = -2.0  # radians; tune as needed

        # Distance at which a lane-clearing rod will drop back down
        # once the ball comes near that rod.
        self.lane_release_dist = 3.0

        # Track which rods are currently "up" for lane clearing.
        # Keys correspond to yellow rods in back-to-front order.
        self.lane_clear_state = {
            "goal": False,
            "def": False,
            "mid": False,
            "attack": False,
        }

        # --- Rod cock state (persists across timesteps) ---
        # Rod remains cocked back until the ball passes its y coordinate.
        self.rod_cocked_state = {
            "goal": False,
            "def": False,
            "mid": False,
            "attack": False,
        }
        self.rod_cocked_y = {
            "goal": 0.0,
            "def": 0.0,
            "mid": 0.0,
            "attack": 0.0,
        }

        # --- Black (antagonist) lane clearing / cock state ---
        self.b_lane_clear_state = {
            "goal": False,
            "def": False,
            "mid": False,
            "attack": False,
        }

        self.b_rod_cocked_state = {
            "goal": False,
            "def": False,
            "mid": False,
            "attack": False,
        }
        self.b_rod_cocked_y = {
            "goal": 0.0,
            "def": 0.0,
            "mid": 0.0,
            "attack": 0.0,
        }

        # If ball approaches a rod from the FRONT fast enough, skip cock-back and only forward-kick.
        # Tune this (table units / sec).
        self.front_fast_vy_thresh = 9.0

        # --- Caching for performance optimization ---
        self._initialize_caches()

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

        # Reset cocked state each episode
        for k in self.rod_cocked_state:
            self.rod_cocked_state[k] = False
            self.rod_cocked_y[k] = 0.0

        # Reset black cocked / lane-clear state each episode
        for k in self.b_rod_cocked_state:
            self.b_rod_cocked_state[k] = False
            self.b_rod_cocked_y[k] = 0.0
        for k in self.b_lane_clear_state:
            self.b_lane_clear_state[k] = False

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
        """
        Build an 8-d action for protagonist (yellow):
          [x_goal, r_goal, x_def, r_def, x_mid, r_mid, x_attack, r_attack]

        - Linear entries (x_*) move each rod toward aligning its closest foosman's X with ball_x.
        - Kicking logic uses the original dx/dy window:
            * cock back when ball is in a "ready" region,
            * forward kick when ball is in a "contact" region.
        - On forward kick:
            * That rod gets a random slide offset (angled shot),
            * All rods IN FRONT of it rotate "up" by clear_lane_angle and STAY up
              until the ball comes close to those rods (lane_release_dist).
        """
        ball_x, ball_y = float(obs[0]), float(obs[1])
        # Physics state is already up-to-date from step() - no need to call mj_forward here

        rods = [
            ("goal",   0, 1, "y_goal_linear",   "y_goal_rotation"),
            ("def",    2, 3, "y_def_linear",    "y_def_rotation"),
            ("mid",    4, 5, "y_mid_linear",    "y_mid_rotation"),
            ("attack", 6, 7, "y_attack_linear", "y_attack_rotation"),
        ]

        action = np.zeros(self.protagonist_action_size, dtype=np.float32)

        ball_vy = float(obs[4])  # ball velocity in Y
        guy_x_map = {}           # NEW: store nearest guy_x per rod

        # First pass: compute per-rod slide/rot targets with kicking + persistent cock state.
        guy_y_map = {}
        rot_lims = {}
        lin_lims = {}
        rot_targets = {}
        slide_targets = {}
        cocked_flags = []

        for rod_key, lin_idx, rot_idx, act_lin, act_rot in rods:
            stats = self._nearest_guy_stats_and_rot_limits("y", rod_key, act_lin, act_rot)
            if stats is None:
                action[lin_idx] = 0.0
                action[rot_idx] = 0.0
                guy_y_map[rod_key] = 0.0
                rot_lims[rod_key] = (-2.5, 2.5)
                lin_lims[rod_key] = (-10.0, 10.0)
                rot_targets[rod_key] = 0.0
                slide_targets[rod_key] = 0.0
                cocked_flags.append(False)
                continue

            (guy_x, guy_y,
             _qpos_adr_rot, rot_min, rot_max,
             _qpos_adr_lin, lin_min, lin_max) = stats

            guy_y_map[rod_key] = guy_y
            guy_x_map[rod_key] = guy_x
            rot_lims[rod_key] = (rot_min, rot_max)
            lin_lims[rod_key] = (lin_min, lin_max)

            dx = abs(ball_x - guy_x)
            dy = ball_y - guy_y  # ball in front of foosman if dy >= 0 for +Y attack

            in_front_ready = (self.kick_y_front_min <= dy <= self.kick_y_front_max) and (dx <= self.kick_x_close)
            in_contact = (0.0 <= dy <= self.kick_y_contact_max) and (dx <= self.kick_x_contact)

            # NEW: detect fast approach from FRONT (from opponent goal toward yellow => ball_vy negative)
            approaching_front_fast = (dy > 0.0) and (ball_vy < -self.front_fast_vy_thresh)

            # Release cocked state once ball passes this rod's Y
            if self.rod_cocked_state[rod_key]:
                if ball_y > self.rod_cocked_y[rod_key]:
                    self.rod_cocked_state[rod_key] = False
                    self.rod_cocked_y[rod_key] = 0.0
            else:
                # Begin cocking only if NOT fast-from-front
                if in_front_ready and (not approaching_front_fast):
                    self.rod_cocked_state[rod_key] = True
                    self.rod_cocked_y[rod_key] = guy_y

            cocked = self.rod_cocked_state[rod_key]
            cocked_flags.append(cocked)

            # Always slide toward ball X
            slide_target = self._slide_target_for_rod_toward_ball("y", rod_key, act_lin, ball_x, ball_y)

            if in_contact:
                slide_target = slide_target + np.random.uniform(-0.5, 0.5)

            # Rotation targets
            if cocked:
                rot_target = self.kick_sign * self.kick_back_angle

                if in_contact:
                    rot_target = self.kick_sign * self.kick_forward_angle
                    slide_target = slide_target + np.random.uniform(-self.kick_offset_max, self.kick_offset_max)
            else:
                rot_target = 0.0
                # NEW: if fast-from-front, skip cocking and only forward-kick at contact
                if approaching_front_fast and in_contact:
                    rot_target = self.kick_sign * self.kick_forward_angle
                    slide_target = slide_target + np.random.uniform(-self.kick_offset_max, self.kick_offset_max)

            # Clamp to ranges
            slide_target = float(np.clip(slide_target, lin_min, lin_max))
            rot_target = float(np.clip(rot_target, rot_min, rot_max))

            slide_targets[rod_key] = slide_target
            rot_targets[rod_key] = rot_target

        # Second pass: lane clearing for rods in front of any kicking rod.
        # Second pass: lane clearing for rods in front of any kicking rod.
        for i, (rod_key, lin_idx, rot_idx, act_lin, act_rot) in enumerate(rods):
            cocked_here = cocked_flags[i]
            any_back_kicking = any(cocked_flags[:i])  # any rod behind is in kicking phase?

            # If this rod is currently lifted, drop once the ball passes its Y
            if self.lane_clear_state[rod_key]:
                if ball_y > guy_y_map.get(rod_key, 0.0):
                    self.lane_clear_state[rod_key] = False

            # RESTORED: if a back rod is kicking, lift this rod (unless it's itself kicking),
            # but don't re-lift if ball already passed it.
            if any_back_kicking and not cocked_here:
                if ball_y <= guy_y_map.get(rod_key, 0.0):
                    self.lane_clear_state[rod_key] = True

            # If lane-clearing rod sees ball coming from BACK, wait and then forward-kick
            if self.lane_clear_state[rod_key] and not cocked_here:
                gx = guy_x_map.get(rod_key, 0.0)
                gy = guy_y_map.get(rod_key, 0.0)
                dx_back = abs(ball_x - gx)
                dy_back = gy - ball_y  # positive when ball is behind yellow foosman
                approaching_from_back = (dy_back > 0.0) and (ball_vy > 0.0)

                if approaching_from_back and (dx_back <= self.kick_x_contact) and (
                        0.0 <= dy_back <= self.kick_y_contact_max):
                    # kick forward instead of holding lane-clear
                    rot_min, rot_max = rot_lims[rod_key]
                    rot_targets[rod_key] = float(np.clip(self.kick_sign * self.kick_forward_angle, rot_min, rot_max))

                    slide_targets[rod_key] = float(
                        np.clip(
                            slide_targets[rod_key] + np.random.uniform(-self.kick_offset_max, self.kick_offset_max),
                            lin_lims[rod_key][0], lin_lims[rod_key][1]
                        )
                    )

                    # NEW: after this rod kicks, drop it back down (restore lane behavior)
                    self.lane_clear_state[rod_key] = False
                else:
                    # normal lane clear hold-up
                    rot_min, rot_max = rot_lims[rod_key]
                    rot_targets[rod_key] = float(np.clip(self.clear_lane_angle, rot_min, rot_max))

            # Write final targets into action vector
            lin_min, lin_max = lin_lims[rod_key]
            action[lin_idx] = float(np.clip(slide_targets[rod_key], lin_min, lin_max))
            action[rot_idx] = float(np.clip(rot_targets[rod_key], rot_lims[rod_key][0], rot_lims[rod_key][1]))

        return action

    def antagonist_all_rods_action_toward_ball(self, obs: np.ndarray) -> np.ndarray:
        """
        Build an 8-d action for antagonist (black):
          [x_goal, r_goal, x_def, r_def, x_mid, r_mid, x_attack, r_attack]

        Black attacks toward -Y (toward yellow goal), so:
          - "ball in front" means ball_y is slightly LESS than guy_y in world coords.
          - A cocked rod stays cocked until ball passes its y going -Y (ball_y < rod_y).
          - Lane clearing lifts rods "in front" toward -Y (lower y), i.e., later in list.
        """
        ball_x, ball_y = float(obs[0]), float(obs[1])
        # Physics state is already up-to-date from step() - no need to call mj_forward here

        # Rod spec in back→front order for black from its own goal (+Y) toward opponent (-Y)
        rods = [
            ("goal",   0, 1, "b_goal_linear",   "b_goal_rotation"),
            ("def",    2, 3, "b_def_linear",    "b_def_rotation"),
            ("mid",    4, 5, "b_mid_linear",    "b_mid_rotation"),
            ("attack", 6, 7, "b_attack_linear", "b_attack_rotation"),
        ]

        action = np.zeros(self.antagonist_action_size, dtype=np.float32)

        ball_vy = float(obs[4])  # ball velocity in Y
        guy_x_map = {}           # NEW

        guy_y_map = {}
        rot_lims = {}
        lin_lims = {}
        rot_targets = {}
        slide_targets = {}
        cocked_flags = []

        # If black rotation direction is opposite, flip sign here.
        # Start with self.kick_sign; if kicks look backwards, change to -self.kick_sign.
        black_kick_sign = self.kick_sign

        for rod_key, lin_idx, rot_idx, act_lin, act_rot in rods:
            stats = self._nearest_guy_stats_and_rot_limits("b", rod_key, act_lin, act_rot)
            if stats is None:
                action[lin_idx] = 0.0
                action[rot_idx] = 0.0
                guy_y_map[rod_key] = 0.0
                rot_lims[rod_key] = (-2.5, 2.5)
                lin_lims[rod_key] = (-10.0, 10.0)
                rot_targets[rod_key] = 0.0
                slide_targets[rod_key] = 0.0
                cocked_flags.append(False)
                continue

            (guy_x, guy_y,
             _qpos_adr_rot, rot_min, rot_max,
             _qpos_adr_lin, lin_min, lin_max) = stats

            guy_y_map[rod_key] = guy_y
            guy_x_map[rod_key] = guy_x
            rot_lims[rod_key] = (rot_min, rot_max)
            lin_lims[rod_key] = (lin_min, lin_max)

            dx = abs(ball_x - guy_x)
            # For black attacking -Y, define "front dy" as guy_y - ball_y
            dy_front = guy_y - ball_y  # positive when ball is in front of black foosman

            in_front_ready = (self.kick_y_front_min <= dy_front <= self.kick_y_front_max) and (dx <= self.kick_x_close)
            in_contact = (0.0 <= dy_front <= self.kick_y_contact_max) and (dx <= self.kick_x_contact)

            # NEW: fast approach from FRONT for black = ball in front AND moving +Y fast
            approaching_front_fast = (dy_front > 0.0) and (ball_vy > self.front_fast_vy_thresh)

            if self.b_rod_cocked_state[rod_key]:
                if ball_y < self.b_rod_cocked_y[rod_key]:
                    self.b_rod_cocked_state[rod_key] = False
                    self.b_rod_cocked_y[rod_key] = 0.0
            else:
                if in_front_ready and (not approaching_front_fast):
                    self.b_rod_cocked_state[rod_key] = True
                    self.b_rod_cocked_y[rod_key] = guy_y

            cocked = self.b_rod_cocked_state[rod_key]
            cocked_flags.append(cocked)

            slide_target = self._slide_target_for_rod_toward_ball("b", rod_key, act_lin, ball_x, ball_y)

            if in_contact:
                slide_target = slide_target + np.random.uniform(-0.5, 0.5)

            if cocked:
                rot_target = black_kick_sign * self.kick_back_angle
                if in_contact:
                    rot_target = black_kick_sign * self.kick_forward_angle
                    slide_target = slide_target + np.random.uniform(-self.kick_offset_max, self.kick_offset_max)
            else:
                rot_target = 0.0
                # NEW: fast-from-front => only forward-kick at contact
                if approaching_front_fast and in_contact:
                    rot_target = black_kick_sign * self.kick_forward_angle
                    slide_target = slide_target + np.random.uniform(-self.kick_offset_max, self.kick_offset_max)

            slide_target = float(np.clip(slide_target, lin_min, lin_max))
            rot_target = float(np.clip(rot_target, rot_min, rot_max))

            slide_targets[rod_key] = slide_target
            rot_targets[rod_key] = rot_target

        # Lane clearing for rods in front of any kicking rod (toward -Y, i.e., later in list)
        for i, (rod_key, lin_idx, rot_idx, act_lin, act_rot) in enumerate(rods):
            cocked_here = cocked_flags[i]
            any_back_kicking = any(cocked_flags[:i])  # rods behind (closer to black goal) kicking?

            # If this rod is lifted, drop once ball passes its Y going -Y
            if self.b_lane_clear_state[rod_key]:
                if ball_y < guy_y_map.get(rod_key, 0.0):
                    self.b_lane_clear_state[rod_key] = False

            # RESTORED: if a back rod is kicking, lift this rod (unless it's itself kicking),
            # but don't re-lift if ball already passed it.
            if any_back_kicking and not cocked_here:
                if ball_y >= guy_y_map.get(rod_key, 0.0):
                    self.b_lane_clear_state[rod_key] = True

            # Lane-clearing rod + ball coming from BACK (for black, back = -Y side)
            if self.b_lane_clear_state[rod_key] and not cocked_here:
                gx = guy_x_map.get(rod_key, 0.0)
                gy = guy_y_map.get(rod_key, 0.0)
                dx_back = abs(ball_x - gx)
                dy_back = ball_y - gy  # positive when ball is behind black foosman
                approaching_from_back = (dy_back > 0.0) and (ball_vy < 0.0)

                if approaching_from_back and (dx_back <= self.kick_x_contact) and (
                        0.0 <= dy_back <= self.kick_y_contact_max):
                    rot_min, rot_max = rot_lims[rod_key]
                    rot_targets[rod_key] = float(np.clip(black_kick_sign * self.kick_forward_angle, rot_min, rot_max))

                    slide_targets[rod_key] = float(
                        np.clip(
                            slide_targets[rod_key] + np.random.uniform(-self.kick_offset_max, self.kick_offset_max),
                            lin_lims[rod_key][0], lin_lims[rod_key][1]
                        )
                    )

                    # NEW: after this rod kicks, drop it back down
                    self.b_lane_clear_state[rod_key] = False
                else:
                    rot_min, rot_max = rot_lims[rod_key]
                    rot_targets[rod_key] = float(np.clip(self.clear_lane_angle, rot_min, rot_max))

            action[lin_idx] = float(np.clip(slide_targets[rod_key], lin_lims[rod_key][0], lin_lims[rod_key][1]))
            action[rot_idx] = float(np.clip(rot_targets[rod_key], rot_lims[rod_key][0], rot_lims[rod_key][1]))

        return action


    # def protagonist_all_rods_action_toward_ball(self, obs: np.ndarray) -> np.ndarray:
    #     """
    #     Build an 8-d action for protagonist (yellow):
    #       [x_goal, r_goal, x_def, r_def, x_mid, r_mid, x_attack, r_attack]
    #
    #     - Linear entries (x_*) move each rod toward aligning its closest foosman's X with ball_x.
    #     - Kicking logic uses the original dx/dy window:
    #         * cock back when ball is in a "ready" region,
    #         * forward kick when ball is in a "contact" region.
    #     - On forward kick:
    #         * That rod gets a random slide offset (angled shot),
    #         * All rods IN FRONT of it rotate "up" by clear_lane_angle and STAY up
    #           until the ball comes close to those rods (lane_release_dist).
    #     """
    #     ball_x, ball_y = float(obs[0]), float(obs[1])
    #     mujoco.mj_forward(self.model, self.data)
    #
    #     # Rod spec in back→front order for yellow
    #     # (rod_key, local_linear_idx, local_rot_idx, actuator_linear_name, actuator_rot_name)
    #     rods = [
    #         ("goal",   0, 1, "y_goal_linear",   "y_goal_rotation"),
    #         ("def",    2, 3, "y_def_linear",    "y_def_rotation"),
    #         ("mid",    4, 5, "y_mid_linear",    "y_mid_rotation"),
    #         ("attack", 6, 7, "y_attack_linear", "y_attack_rotation"),
    #     ]
    #
    #     action = np.zeros(self.protagonist_action_size, dtype=np.float32)
    #
    #     # 1) Linear targets + nearest-guy stats for each rod
    #     #    Tuple:
    #     #      (order_idx, rod_key, lin_i, rot_i,
    #     #       dx, dy, d2, guy_x, guy_y,
    #     #       qpos_adr_rot, rot_min, rot_max,
    #     #       qpos_adr_lin, lin_min, lin_max)
    #     nearest_stats = []
    #     # slide all rods towards ball
    #     for order_idx, (rod_key, lin_i, rot_i, act_lin, act_rot) in enumerate(rods):
    #         # Slide toward ball (using your aggressive slide logic helper)
    #         s_target = self._slide_target_for_rod_toward_ball(
    #             team="y",
    #             rod_key=rod_key,
    #             actuator_name=act_lin,
    #             ball_x=ball_x,
    #             ball_y=ball_y,
    #         )
    #         action[lin_i] = s_target
    #
    #         stats = self._nearest_guy_stats_and_rot_limits(
    #             team="y",
    #             rod_key=rod_key,
    #             actuator_lin_name=act_lin,
    #             actuator_rot_name=act_rot,
    #         )
    #         if stats is None:
    #             continue
    #
    #         (guy_x, guy_y,
    #          qpos_adr_rot, rot_min, rot_max,
    #          qpos_adr_lin, lin_min, lin_max) = stats
    #
    #         dx = abs(ball_x - guy_x)
    #         dy = ball_y - guy_y
    #         d2 = dx * dx + dy * dy
    #
    #         nearest_stats.append(
    #             (order_idx, rod_key, lin_i, rot_i,
    #              dx, dy, d2, guy_x, guy_y,
    #              qpos_adr_rot, rot_min, rot_max,
    #              qpos_adr_lin, lin_min, lin_max)
    #         )
    #
    #     if not nearest_stats:
    #         return action
    #
    #     for (_ord_i, rod_key_i, _lin_i, _rot_i,
    #          _dx_i, _dy_i, _d2_i, _gx_i, _gy_i,
    #          _qrot_i, _rmin_i, _rmax_i,
    #          _qlin_i, _lmin_i, _lmax_i) in nearest_stats:
    #         if self.rod_cocked_state.get(rod_key_i, False):
    #             # Attack direction assumed +Y: once ball_y > stored rod_y, it's passed
    #             if ball_y > self.rod_cocked_y.get(rod_key_i, 0.0):
    #                 self.rod_cocked_state[rod_key_i] = False
    #
    #     # 2) Shooter = rod whose nearest guy is closest to the ball
    #     (order_idx_sel, rod_key_sel, lin_i_sel, rot_i_sel,
    #      dx_sel, dy_sel, d2_sel, guy_x_sel, guy_y_sel,
    #      qpos_adr_rot_sel, rot_min_sel, rot_max_sel,
    #      qpos_adr_lin_sel, lin_min_sel, lin_max_sel) = min(nearest_stats, key=lambda t: t[6])
    #
    #     # 3) Update lane-clear state: drop rods that the ball has come near
    #     for (ord_i, rod_key_i, _lin_i, _rot_i,
    #          _dx_i, _dy_i, d2_i, _gx_i, _gy_i,
    #          _qrot_i, _rmin_i, _rmax_i,
    #          _qlin_i, _lmin_i, _lmax_i) in nearest_stats:
    #         if self.lane_clear_state.get(rod_key_i, False):
    #             dist_i = math.sqrt(d2_i)
    #             if dist_i <= self.lane_release_dist:
    #                 self.lane_clear_state[rod_key_i] = False
    #
    #     # 4) Old kicking logic based on dx, dy windows + persistent cocked-back state
    #     r_target = None
    #     did_forward_kick = False
    #
    #     # If this rod is already cocked and the ball hasn't passed its stored y yet,
    #     # keep it cocked back.
    #     if self.rod_cocked_state.get(rod_key_sel, False) and \
    #        ball_y <= self.rod_cocked_y.get(rod_key_sel, guy_y_sel):
    #         r_target = self.kick_sign * self.kick_back_angle
    #     else:
    #         # Cock back: ball near in X and slightly in front in Y
    #         if (dx_sel <= self.kick_x_close) and (self.kick_y_front_min <= dy_sel <= self.kick_y_front_max):
    #             r_target = self.kick_sign * self.kick_back_angle
    #             # Mark this rod as cocked and store its y position at cock time
    #             self.rod_cocked_state[rod_key_sel] = True
    #             self.rod_cocked_y[rod_key_sel] = guy_y_sel
    #
    #         # Forward kick: ball very close in X and almost aligned in Y
    #         if (dx_sel <= self.kick_x_contact) and (0.0 <= dy_sel <= self.kick_y_contact_max):
    #             r_target = -self.kick_sign * self.kick_forward_angle  # override back-cock
    #             did_forward_kick = True
    #             # Forward kick ends the cocked state
    #             self.rod_cocked_state[rod_key_sel] = False
    #
    #     # Apply shooter rotation (if any)
    #     if r_target is not None:
    #         r_target = float(np.clip(r_target, rot_min_sel, rot_max_sel))
    #         action[rot_i_sel] = r_target
    #
    #
    #     # # 4) Old kicking logic based on dx, dy windows
    #     # r_target = None
    #     # did_forward_kick = False
    #     #
    #     # # Cock back: ball near in X and slightly in front in Y
    #     # if (dx_sel <= self.kick_x_close) and (self.kick_y_front_min <= dy_sel <= self.kick_y_front_max):
    #     #     r_target = self.kick_sign * self.kick_back_angle
    #     #     did_forward_kick = True
    #     #
    #     # # Forward kick: ball very close in X and almost aligned in Y
    #     # if (dx_sel <= self.kick_x_contact) and (0.0 <= dy_sel <= self.kick_y_contact_max):
    #     #     r_target = -self.kick_sign * self.kick_forward_angle  # override back-cock
    #     #     did_forward_kick = True
    #     #
    #     # # Apply shooter rotation (if any)
    #     # if r_target is not None:
    #     #     r_target = float(np.clip(r_target, rot_min_sel, rot_max_sel))
    #     #     action[rot_i_sel] = r_target
    #
    #     # 5) On forward kick:
    #     #    - random slide offset for the shooter (angle shot)
    #     #    - mark rods in front for lane clearing
    #     if did_forward_kick:
    #         # 5a) Random offset on slide for shooter
    #         rng = getattr(self, "np_random", np.random)
    #         offset = rng.uniform(-self.kick_offset_max, self.kick_offset_max)
    #
    #         current_slide = float(self.data.qpos[qpos_adr_lin_sel])
    #         slide_target = current_slide + offset
    #         slide_target = float(np.clip(slide_target, lin_min_sel, lin_max_sel))
    #         action[lin_i_sel] = slide_target  # overwrite earlier slide
    #
    #         # 5b) Mark rods IN FRONT of shooter for lane clearing
    #         for (ord_i, rod_key_i, _lin_i, _rot_i,
    #              _dx_i, _dy_i, _d2_i, _gx_i, _gy_i,
    #              _qrot_i, _rmin_i, _rmax_i,
    #              _qlin_i, _lmin_i, _lmax_i) in nearest_stats:
    #             if ord_i > order_idx_sel:
    #                 self.lane_clear_state[rod_key_i] = True
    #
    #     # # 6) Apply lane-clearing rotations for rods that are up (except shooter)
    #     # clear_angle = self.kick_sign * self.clear_lane_angle
    #     # for (ord_i, rod_key_i, _lin_i, rot_i,
    #     #      _dx_i, _dy_i, _d2_i, _gx_i, _gy_i,
    #     #      _qrot_i, rmin_i, rmax_i,
    #     #      _qlin_i, _lmin_i, _lmax_i) in nearest_stats:
    #     #
    #     #     if rod_key_i == rod_key_sel:
    #     #         continue  # don't override shooter rotation
    #     #
    #     #     if self.lane_clear_state.get(rod_key_i, False):
    #     #         r_clear = float(np.clip(clear_angle, rmin_i, rmax_i))
    #     #         action[rot_i] = r_clear
    #
    #     # 6) Apply lane-clearing rotations for rods that are up (except shooter)
    #     clear_angle = self.kick_sign * self.clear_lane_angle
    #     for (ord_i, rod_key_i, _lin_i, rot_i,
    #          _dx_i, _dy_i, _d2_i, _gx_i, _gy_i,
    #          _qrot_i, rmin_i, rmax_i,
    #          _qlin_i, _lmin_i, _lmax_i) in nearest_stats:
    #
    #         if rod_key_i == rod_key_sel:
    #             continue  # don't override shooter rotation
    #
    #         # Lane clearing (existing behavior)
    #         if self.lane_clear_state.get(rod_key_i, False):
    #             r_clear = float(np.clip(clear_angle, rmin_i, rmax_i))
    #             action[rot_i] = r_clear
    #
    #         # Persistent cocked-back rotation for this rod (if still cocked and ball not passed)
    #         if self.rod_cocked_state.get(rod_key_i, False) and \
    #            ball_y <= self.rod_cocked_y.get(rod_key_i, _gy_i):
    #             r_cocked = float(np.clip(self.kick_sign * self.kick_back_angle, rmin_i, rmax_i))
    #             action[rot_i] = r_cocked
    #
    #
    #     return action


    # ------------------------ helpers (private) ------------------------

    # def _slide_target_for_rod_toward_ball(self, team: str, rod_key: str, actuator_name: str,
    #                                       ball_x: float, ball_y: float) -> float:
    #     """
    #     Desired SLIDE setpoint so the closest foosman on the rod aligns its world-X with ball_x.
    #     """
    #     act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    #     if act_id < 0:
    #         return 0.0
    #
    #     jnt_id = int(self.model.actuator_trnid[act_id, 0])
    #     qpos_adr = int(self.model.jnt_qposadr[jnt_id])
    #     ctrl_min = float(self.model.actuator_ctrlrange[act_id, 0])
    #     ctrl_max = float(self.model.actuator_ctrlrange[act_id, 1])
    #
    #     guy_bids = self._rod_guy_body_ids(team, rod_key)
    #     if not guy_bids:
    #         return 0.0
    #
    #     mujoco.mj_forward(self.model, self.data)
    #     men_xy = np.array([[self.data.xpos[bid, 0], self.data.xpos[bid, 1]] for bid in guy_bids], dtype=np.float64)
    #     d2 = (men_xy[:, 0] - ball_x) ** 2 + (men_xy[:, 1] - ball_y) ** 2
    #     idx_closest = int(np.argmin(d2))
    #     guy_x = float(men_xy[idx_closest, 0])
    #
    #     s0 = float(self.data.qpos[qpos_adr])
    #     dx_ds = self._finite_diff_guy_x_wrt_slide(qpos_adr, guy_bids, idx_closest)
    #
    #     if abs(dx_ds) < 1e-6:
    #         direction = np.sign(ball_x - guy_x) if ball_x != guy_x else 0.0
    #         s_target = s0 + 1.0 * direction
    #     else:
    #         s_target = s0 + (ball_x - guy_x) / dx_ds
    #
    #     return float(np.clip(s_target, ctrl_min, ctrl_max))

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
