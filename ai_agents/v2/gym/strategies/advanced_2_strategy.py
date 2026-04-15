"""Advanced2Strategy — real competitive foosball principles.

Key mechanics:
  - **Pull shot** (attack): lateral slide THEN forward kick (L-shaped motion)
  - **Two-bar defense**: goalie + defense stagger for maximum coverage
  - **Fast clears**: max power, wall-aimed, speed over precision
  - **Wall passes**: kick along wall rail for reliable passing
  - **Per-rod state machine** with explicit enum states
"""

import enum
import numpy as np

from ai_agents.v2.gym.strategies.smart_base import (
    SmartFoosballBase, ROD_KEYS, DEFENDING, NEUTRAL, ATTACKING,
)


class RodState(enum.IntEnum):
    IDLE = 0
    BLOCKING = 1
    STRIKING = 3
    LANE_CLEAR = 6


class Advanced2Strategy(SmartFoosballBase):

    def __init__(self, env, team: str):
        super().__init__(env, team)

        # --- General kick params ---
        self.kick_strike_speed = 2.5       # max forward rotation for strikes
        self.goalie_blocking_angle = 0.4   # toes-forward posture for goalie
        self.def_blocking_angle = -0.3     # toes-backward posture for defense

        # --- Two-bar defense ---
        self.stagger_amount = 5.0          # slide offset between goalie and def
        self.goalie_center_bias = 0.5      # how much goalie biases toward center

        # --- "In reach" zone: slightly in front of rod => always kick ---
        self.reach_dx = 2.0               # X-distance for "in reach"
        self.reach_dy_front_max = 3.0     # max front Y-distance for "in reach"
        self.reach_dy_front_min = -0.5    # allow slightly behind (ball rolling past)

        # --- Behind-approach reach: ball coming from own side toward rod ---
        self.reach_dy_back_max = 4.0      # Y-distance behind rod for behind-approach kick

        # --- Per-rod state ---
        self.rod_state = {k: RodState.IDLE for k in ROD_KEYS}

        # --- Lane clearing ---
        self.lane_clear_state = {k: False for k in ROD_KEYS}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def reset(self):
        for k in ROD_KEYS:
            self.rod_state[k] = RodState.IDLE
            self.lane_clear_state[k] = False

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        ball_x, ball_y = float(obs[0]), float(obs[1])
        ball_vx, ball_vy = float(obs[3]), float(obs[4])

        phase = self._detect_phase(ball_y, ball_vy)

        prefix = self.team
        rods = [
            ("goal",   0, 1, f"{prefix}_goal_linear",   f"{prefix}_goal_rotation"),
            ("def",    2, 3, f"{prefix}_def_linear",     f"{prefix}_def_rotation"),
            ("mid",    4, 5, f"{prefix}_mid_linear",     f"{prefix}_mid_rotation"),
            ("attack", 6, 7, f"{prefix}_attack_linear",  f"{prefix}_attack_rotation"),
        ]

        action_size = self.env.protagonist_action_size
        action = np.zeros(action_size, dtype=np.float32)

        guy_y_map = {}
        guy_x_map = {}
        rot_lims = {}
        lin_lims = {}
        rod_stats = {}
        kicking_flags = []  # True if rod is in STRIKING or CLEARING

        # Gather per-rod info
        for rod_key, lin_idx, rot_idx, act_lin, act_rot in rods:
            stats = self.env._nearest_guy_stats_and_rot_limits(prefix, rod_key, act_lin, act_rot)
            rod_stats[rod_key] = stats
            if stats is None:
                guy_y_map[rod_key] = 0.0
                guy_x_map[rod_key] = 0.0
                rot_lims[rod_key] = (-2.5, 2.5)
                lin_lims[rod_key] = (-10.0, 10.0)
            else:
                (gx, gy, _qadr_r, rmin, rmax, _qadr_l, lmin, lmax) = stats
                guy_y_map[rod_key] = gy
                guy_x_map[rod_key] = gx
                rot_lims[rod_key] = (rmin, rmax)
                lin_lims[rod_key] = (lmin, lmax)

        # Dispatch per-rod logic
        slide_targets = {}
        rot_targets = {}
        for rod_key, lin_idx, rot_idx, act_lin, act_rot in rods:
            if rod_stats[rod_key] is None:
                slide_targets[rod_key] = 0.0
                rot_targets[rod_key] = 0.0
                kicking_flags.append(False)
                continue

            gx = guy_x_map[rod_key]
            gy = guy_y_map[rod_key]

            if rod_key == "goal":
                s, r = self._goal_rod_logic(
                    rod_key, act_lin, obs, phase, gx, gy)
            elif rod_key == "def":
                s, r = self._def_rod_logic(
                    rod_key, act_lin, obs, phase, gx, gy)
            elif rod_key == "mid":
                s, r = self._mid_rod_logic(
                    rod_key, act_lin, obs, phase, gx, gy)
            else:  # attack
                s, r = self._attack_rod_logic(
                    rod_key, act_lin, obs, phase, gx, gy)

            slide_targets[rod_key] = s
            rot_targets[rod_key] = r
            kicking_flags.append(self.rod_state[rod_key] == RodState.STRIKING)

        # Two-bar stagger post-processing
        self._apply_two_bar_stagger(
            slide_targets, guy_x_map, ball_x)

        # Lane clearing
        for i, (rod_key, lin_idx, rot_idx, act_lin, act_rot) in enumerate(rods):
            any_back_kicking = any(kicking_flags[:i])
            kicking_here = kicking_flags[i]

            if self.lane_clear_state[rod_key]:
                if self._ball_passed_rod(ball_y, guy_y_map.get(rod_key, 0.0)):
                    self.lane_clear_state[rod_key] = False

            if any_back_kicking and not kicking_here:
                if not self._ball_passed_rod(ball_y, guy_y_map.get(rod_key, 0.0)):
                    self.lane_clear_state[rod_key] = True

            if self.lane_clear_state[rod_key] and not kicking_here:
                rmin, rmax = rot_lims[rod_key]
                rot_targets[rod_key] = float(np.clip(self.clear_lane_angle, rmin, rmax))
                self.rod_state[rod_key] = RodState.LANE_CLEAR

        # Write final actions with clipping
        for rod_key, lin_idx, rot_idx, act_lin, act_rot in rods:
            rmin, rmax = rot_lims[rod_key]
            lmin, lmax = lin_lims[rod_key]
            action[lin_idx] = float(np.clip(slide_targets[rod_key], lmin, lmax))
            action[rot_idx] = float(np.clip(rot_targets[rod_key], rmin, rmax))

        return action

    # ------------------------------------------------------------------ #
    # Goal rod — Two-bar defense (blocking half)
    # ------------------------------------------------------------------ #

    def _goal_rod_logic(self, rod_key, act_lin, obs, phase, gx, gy):
        ball_x, ball_y = float(obs[0]), float(obs[1])
        ball_vx, ball_vy = float(obs[3]), float(obs[4])

        # Ball in reach => always kick immediately
        if self._ball_in_reach(ball_x, ball_y, gx, gy):
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, ball_x, ball_y)
            self.rod_state[rod_key] = RodState.STRIKING
            return slide_target, self.kick_sign * self.kick_strike_speed

        state = self.rod_state[rod_key]

        # Transition: ball passed rod => back to IDLE
        if self._ball_passed_rod(ball_y, gy):
            state = RodState.IDLE

        # IDLE / BLOCKING logic
        if state in (RodState.IDLE, RodState.BLOCKING, RodState.LANE_CLEAR, RodState.STRIKING):
            if phase == DEFENDING or self._is_ball_approaching(ball_vy):
                state = RodState.BLOCKING
            else:
                state = RodState.IDLE

        # Compute outputs based on state
        if state == RodState.BLOCKING:
            # Prediction-based slide
            pred_x = self._predict_arrival_x(ball_x, ball_y, ball_vx, ball_vy, gy)
            base = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, pred_x, ball_y)
            slide_target = base * (1 - self.goalie_center_bias) + 0.0 * self.goalie_center_bias
            rot_target = self.kick_sign * self.goalie_blocking_angle

        else:  # IDLE
            # Center-biased slide, blocking posture
            base = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, ball_x, ball_y)
            slide_target = base * (1 - self.goalie_center_bias) + 0.0 * self.goalie_center_bias
            rot_target = self.kick_sign * self.goalie_blocking_angle

        self.rod_state[rod_key] = state
        return slide_target, rot_target

    # ------------------------------------------------------------------ #
    # Defense rod — Two-bar defense (other half)
    # ------------------------------------------------------------------ #

    def _def_rod_logic(self, rod_key, act_lin, obs, phase, gx, gy):
        ball_x, ball_y = float(obs[0]), float(obs[1])
        ball_vx, ball_vy = float(obs[3]), float(obs[4])

        # Ball in reach => always kick immediately
        if self._ball_in_reach(ball_x, ball_y, gx, gy):
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, ball_x, ball_y)
            self.rod_state[rod_key] = RodState.STRIKING
            return slide_target, self.kick_sign * self.kick_strike_speed

        state = self.rod_state[rod_key]

        # Transition: ball passed rod => IDLE
        if self._ball_passed_rod(ball_y, gy):
            state = RodState.IDLE

        # IDLE / BLOCKING
        if state in (RodState.IDLE, RodState.BLOCKING, RodState.LANE_CLEAR, RodState.STRIKING):
            if phase == DEFENDING or self._is_ball_approaching(ball_vy):
                state = RodState.BLOCKING
            else:
                state = RodState.IDLE

        # Compute outputs
        if state == RodState.BLOCKING:
            pred_x = self._predict_arrival_x(ball_x, ball_y, ball_vx, ball_vy, gy)
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, pred_x, ball_y)
            rot_target = self.kick_sign * self.def_blocking_angle

        else:  # IDLE
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, ball_x, ball_y)
            rot_target = 0.0

        self.rod_state[rod_key] = state
        return slide_target, rot_target

    # ------------------------------------------------------------------ #
    # Midfield rod — Transition / passing
    # ------------------------------------------------------------------ #

    def _mid_rod_logic(self, rod_key, act_lin, obs, phase, gx, gy):
        ball_x, ball_y = float(obs[0]), float(obs[1])
        ball_vx, ball_vy = float(obs[3]), float(obs[4])

        # Ball in reach => always kick immediately
        if self._ball_in_reach(ball_x, ball_y, gx, gy):
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, ball_x, ball_y)
            self.rod_state[rod_key] = RodState.STRIKING
            return slide_target, self.kick_sign * self.kick_strike_speed

        state = self.rod_state[rod_key]

        # Transition: ball passed rod => IDLE
        if self._ball_passed_rod(ball_y, gy):
            state = RodState.IDLE

        # IDLE / BLOCKING
        if state in (RodState.IDLE, RodState.BLOCKING, RodState.LANE_CLEAR, RodState.STRIKING):
            if self._is_ball_approaching(ball_vy):
                state = RodState.BLOCKING
            else:
                state = RodState.IDLE

        # Compute outputs
        if state == RodState.BLOCKING:
            pred_x = self._predict_arrival_x(ball_x, ball_y, ball_vx, ball_vy, gy)
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, pred_x, ball_y)
            rot_target = self.kick_sign * self.def_blocking_angle

        else:  # IDLE
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, ball_x, ball_y)
            rot_target = 0.0

        self.rod_state[rod_key] = state
        return slide_target, rot_target

    # ------------------------------------------------------------------ #
    # Attack rod — Pull shot
    # ------------------------------------------------------------------ #

    def _attack_rod_logic(self, rod_key, act_lin, obs, phase, gx, gy):
        ball_x, ball_y = float(obs[0]), float(obs[1])
        ball_vx, ball_vy = float(obs[3]), float(obs[4])

        # Ball in reach => always kick immediately (overrides pull shot)
        if self._ball_in_reach(ball_x, ball_y, gx, gy):
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, ball_x, ball_y)
            self.rod_state[rod_key] = RodState.STRIKING
            self.pull_steps = 0
            return slide_target, self.kick_sign * self.kick_strike_speed

        state = self.rod_state[rod_key]

        # Transition: ball passed rod => IDLE
        if self._ball_passed_rod(ball_y, gy):
            state = RodState.IDLE
            self.pull_steps = 0

        # ---- State transitions ----

        if state in (RodState.IDLE, RodState.BLOCKING, RodState.LANE_CLEAR, RodState.STRIKING):
            if self._is_ball_approaching(ball_vy):
                state = RodState.BLOCKING
            else:
                state = RodState.IDLE

        # Compute outputs

        if state == RodState.BLOCKING:
            # Predict and track
            pred_x = self._predict_arrival_x(ball_x, ball_y, ball_vx, ball_vy, gy)
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, pred_x, ball_y)
            rot_target = 0.0  # men down ready to receive

        else:  # IDLE
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, ball_x, ball_y)
            rot_target = 0.0

        self.rod_state[rod_key] = state
        return slide_target, rot_target

    # ------------------------------------------------------------------ #
    # Two-bar stagger post-processing
    # ------------------------------------------------------------------ #

    def _apply_two_bar_stagger(self, slide_targets, guy_x_map, ball_x):
        """Offset defense rod slide relative to goalie for better coverage.

        When both goal and def are in BLOCKING state, offset the defense
        rod so its foosmen interleave with the goalie's.
        """
        goal_blocking = self.rod_state["goal"] in (RodState.BLOCKING, RodState.IDLE)
        def_blocking = self.rod_state["def"] in (RodState.BLOCKING, RodState.IDLE)

        if goal_blocking and def_blocking:
            # Offset defense rod opposite to ball X direction
            if ball_x > 0:
                slide_targets["def"] -= self.stagger_amount
            else:
                slide_targets["def"] += self.stagger_amount

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _ball_in_reach(self, ball_x, ball_y, gx, gy):
        """True when ball is within the 'always kick' zone slightly in front of the rod."""
        dx = abs(ball_x - gx)
        dy_front = self._dy_front(ball_y, gy)
        return (dx <= self.reach_dx and
                self.reach_dy_front_min <= dy_front <= self.reach_dy_front_max)

    def _ball_behind_and_close(self, ball_x, ball_y, ball_vy, gx, gy):
        """True when ball is behind the rod, approaching from behind, and close enough to kick."""
        if not self._is_approaching_from_back(ball_vy):
            return False
        dx = abs(ball_x - gx)
        dy_back = self._dy_back(ball_y, gy)
        return dx <= self.reach_dx and 0 < dy_back <= self.reach_dy_back_max

