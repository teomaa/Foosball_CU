"""BasicStrategy — direct extraction of the current rule-based logic.

Unifies the protagonist (yellow, +Y) and antagonist (black, -Y) action methods
into a single ``compute_action`` by using a *direction* sign that flips the
relevant dy/vy comparisons.
"""

import numpy as np

from ai_agents.v2.gym.strategies.base_strategy import FoosballStrategy

ROD_KEYS = ["goal", "def", "mid", "attack"]


class BasicStrategy(FoosballStrategy):

    def __init__(self, env, team: str):
        super().__init__(env, team)

        # --- Kick params ---
        self.kick_x_close = 2.0
        self.kick_x_contact = 0.3
        self.kick_y_front_min = -1.2
        self.kick_y_front_max = 9.0
        self.kick_y_contact_max = 1.25
        self.kick_back_angle = 1.95
        self.kick_forward_angle = 3.90
        self.kick_sign = 1.0
        self.kick_offset_max = 5.0

        # --- Lane clearing ---
        self.clear_lane_angle = -2.0
        self.lane_release_dist = 3.0

        # --- Fast approach threshold ---
        self.front_fast_vy_thresh = 9.0

        # --- Persistent state ---
        self.rod_cocked_state = {k: False for k in ROD_KEYS}
        self.rod_cocked_y = {k: 0.0 for k in ROD_KEYS}
        self.lane_clear_state = {k: False for k in ROD_KEYS}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def reset(self):
        for k in ROD_KEYS:
            self.rod_cocked_state[k] = False
            self.rod_cocked_y[k] = 0.0
            self.lane_clear_state[k] = False

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        ball_x, ball_y = float(obs[0]), float(obs[1])
        ball_vy = float(obs[4])

        prefix = self.team  # "y" or "b"
        d = self.direction  # +1 yellow, -1 black

        rods = [
            ("goal",   0, 1, f"{prefix}_goal_linear",   f"{prefix}_goal_rotation"),
            ("def",    2, 3, f"{prefix}_def_linear",     f"{prefix}_def_rotation"),
            ("mid",    4, 5, f"{prefix}_mid_linear",     f"{prefix}_mid_rotation"),
            ("attack", 6, 7, f"{prefix}_attack_linear",  f"{prefix}_attack_rotation"),
        ]

        action_size = self.env.protagonist_action_size  # same for both teams
        action = np.zeros(action_size, dtype=np.float32)

        guy_x_map = {}
        guy_y_map = {}
        rot_lims = {}
        lin_lims = {}
        rot_targets = {}
        slide_targets = {}
        cocked_flags = []

        kick_sign = self.kick_sign

        # ---- First pass: per-rod slide/rot with cocking ---- #
        for rod_key, lin_idx, rot_idx, act_lin, act_rot in rods:
            stats = self.env._nearest_guy_stats_and_rot_limits(prefix, rod_key, act_lin, act_rot)
            if stats is None:
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
            dy_front = self._dy_front(ball_y, guy_y)

            in_front_ready = (self.kick_y_front_min <= dy_front <= self.kick_y_front_max) and (dx <= self.kick_x_close)
            in_contact = (0.0 <= dy_front <= self.kick_y_contact_max) and (dx <= self.kick_x_contact)
            approaching_front_fast = (dy_front > 0.0) and self._is_fast_approach(ball_vy)

            # Persistent cock state
            if self.rod_cocked_state[rod_key]:
                if self._ball_passed_rod(ball_y, self.rod_cocked_y[rod_key]):
                    self.rod_cocked_state[rod_key] = False
                    self.rod_cocked_y[rod_key] = 0.0
            else:
                if in_front_ready and (not approaching_front_fast):
                    self.rod_cocked_state[rod_key] = True
                    self.rod_cocked_y[rod_key] = guy_y

            cocked = self.rod_cocked_state[rod_key]
            cocked_flags.append(cocked)

            # Slide target
            slide_target = self.env._slide_target_for_rod_toward_ball(prefix, rod_key, act_lin, ball_x, ball_y)
            if in_contact:
                slide_target += np.random.uniform(-0.5, 0.5)

            # Rotation target
            if cocked:
                rot_target = kick_sign * self.kick_back_angle
                if in_contact:
                    rot_target = kick_sign * self.kick_forward_angle
                    slide_target += np.random.uniform(-self.kick_offset_max, self.kick_offset_max)
            else:
                rot_target = 0.0
                if approaching_front_fast and in_contact:
                    rot_target = kick_sign * self.kick_forward_angle
                    slide_target += np.random.uniform(-self.kick_offset_max, self.kick_offset_max)

            slide_target = float(np.clip(slide_target, lin_min, lin_max))
            rot_target = float(np.clip(rot_target, rot_min, rot_max))

            slide_targets[rod_key] = slide_target
            rot_targets[rod_key] = rot_target

        # ---- Second pass: lane clearing ---- #
        for i, (rod_key, lin_idx, rot_idx, act_lin, act_rot) in enumerate(rods):
            cocked_here = cocked_flags[i]
            any_back_kicking = any(cocked_flags[:i])

            # Drop lane-clear once ball passes this rod
            if self.lane_clear_state[rod_key]:
                if self._ball_passed_rod(ball_y, guy_y_map.get(rod_key, 0.0)):
                    self.lane_clear_state[rod_key] = False

            # Lift rod if a back rod is kicking and ball hasn't passed yet
            if any_back_kicking and not cocked_here:
                if not self._ball_passed_rod(ball_y, guy_y_map.get(rod_key, 0.0)):
                    self.lane_clear_state[rod_key] = True

            # Lane-clearing rod: check for ball approaching from back -> forward kick
            if self.lane_clear_state[rod_key] and not cocked_here:
                gx = guy_x_map.get(rod_key, 0.0)
                gy = guy_y_map.get(rod_key, 0.0)
                dx_back = abs(ball_x - gx)
                dy_back = self._dy_back(ball_y, gy)
                approaching_from_back = (dy_back > 0.0) and self._is_approaching_from_back(ball_vy)

                if approaching_from_back and (dx_back <= self.kick_x_contact) and (
                        0.0 <= dy_back <= self.kick_y_contact_max):
                    rot_min, rot_max = rot_lims[rod_key]
                    rot_targets[rod_key] = float(np.clip(kick_sign * self.kick_forward_angle, rot_min, rot_max))
                    slide_targets[rod_key] = float(
                        np.clip(
                            slide_targets[rod_key] + np.random.uniform(-self.kick_offset_max, self.kick_offset_max),
                            lin_lims[rod_key][0], lin_lims[rod_key][1]
                        )
                    )
                    self.lane_clear_state[rod_key] = False
                else:
                    rot_min, rot_max = rot_lims[rod_key]
                    rot_targets[rod_key] = float(np.clip(self.clear_lane_angle, rot_min, rot_max))

            # Write to action
            action[lin_idx] = float(np.clip(slide_targets[rod_key], lin_lims[rod_key][0], lin_lims[rod_key][1]))
            action[rot_idx] = float(np.clip(rot_targets[rod_key], rot_lims[rod_key][0], rot_lims[rod_key][1]))

        return action

    # ------------------------------------------------------------------ #
    # Direction helpers
    # ------------------------------------------------------------------ #

    def _dy_front(self, ball_y: float, guy_y: float) -> float:
        """Positive when ball is *in front* of the foosman (toward opponent goal)."""
        if self.direction == 1:  # yellow attacks +Y
            return ball_y - guy_y
        else:  # black attacks -Y
            return guy_y - ball_y

    def _dy_back(self, ball_y: float, guy_y: float) -> float:
        """Positive when ball is *behind* the foosman (toward own goal)."""
        if self.direction == 1:  # yellow
            return guy_y - ball_y
        else:  # black
            return ball_y - guy_y

    def _ball_passed_rod(self, ball_y: float, rod_y: float) -> bool:
        """Has the ball crossed *past* the rod in the attack direction?"""
        if self.direction == 1:  # yellow attacks +Y
            return ball_y > rod_y
        else:  # black attacks -Y
            return ball_y < rod_y

    def _is_fast_approach(self, ball_vy: float) -> bool:
        """Ball approaching fast from the *front* (from opponent side)."""
        if self.direction == 1:  # yellow: opponent is +Y, fast approach = vy < -thresh
            return ball_vy < -self.front_fast_vy_thresh
        else:  # black: opponent is -Y, fast approach = vy > +thresh
            return ball_vy > self.front_fast_vy_thresh

    def _is_approaching_from_back(self, ball_vy: float) -> bool:
        """Ball moving in the attack direction (coming from behind toward front)."""
        if self.direction == 1:  # yellow attacks +Y => ball moving +Y
            return ball_vy > 0.0
        else:  # black attacks -Y => ball moving -Y
            return ball_vy < 0.0
