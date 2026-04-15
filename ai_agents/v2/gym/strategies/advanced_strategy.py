"""AdvancedStrategy — smarter rule-based agent implementing PROPOSED_RULES.md.

Key improvements over BasicStrategy:
  - Game-phase detection (DEFENDING / NEUTRAL / ATTACKING)
  - Role-specific behavior for each rod (goalkeeper, defense, midfield, attack)
  - Ball trajectory prediction with wall-bounce reflection
  - Opponent gap analysis for aimed shots
  - Deliberate pass targeting toward next rod's foosman
"""

import numpy as np

from ai_agents.v2.gym.strategies.smart_base import (
    SmartFoosballBase, ROD_KEYS, DEFENDING, NEUTRAL, ATTACKING,
)


class AdvancedStrategy(SmartFoosballBase):

    def __init__(self, env, team: str):
        super().__init__(env, team)

        # --- 2-phase kick params (AdvancedStrategy-specific) ---
        self.kick_back_speed_gain = 0.02
        self.kick_back_default = 0.5
        self.kick_back_min_speed = 0.5
        self.kick_strike_speed = 10.0
        self.cock_engage_dy_max = 3.0
        self.kick_offset_max = 5.0

        # --- Goalkeeper params ---
        self.goalie_blocking_angle = 0.4
        self.goalie_center_bias = 0.6
        self.goalie_clear_angle = 5.0

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
        cocked_flags = []

        # Gather per-rod info
        rod_stats = {}
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

        # Dispatch per-rod role
        for rod_key, lin_idx, rot_idx, act_lin, act_rot in rods:
            if rod_stats[rod_key] is None:
                action[lin_idx] = 0.0
                action[rot_idx] = 0.0
                cocked_flags.append(False)
                continue

            if rod_key == "goal":
                s, r, cocked = self._goalkeeper_action(
                    rod_key, act_lin, obs, phase, rod_stats[rod_key])
            elif rod_key == "def":
                s, r, cocked = self._defense_action(
                    rod_key, act_lin, obs, phase, rod_stats[rod_key])
            elif rod_key == "mid":
                s, r, cocked = self._midfield_action(
                    rod_key, act_lin, obs, phase, rod_stats[rod_key])
            else:  # attack
                s, r, cocked = self._attack_action(
                    rod_key, act_lin, obs, phase, rod_stats[rod_key])

            rmin, rmax = rot_lims[rod_key]
            lmin, lmax = lin_lims[rod_key]
            action[lin_idx] = float(np.clip(s, lmin, lmax))
            action[rot_idx] = float(np.clip(r, rmin, rmax))
            cocked_flags.append(cocked)

        # Lane clearing pass
        for i, (rod_key, lin_idx, rot_idx, act_lin, act_rot) in enumerate(rods):
            any_back_kicking = any(cocked_flags[:i])
            cocked_here = cocked_flags[i]

            if self.lane_clear_state[rod_key]:
                if self._ball_passed_rod(ball_y, guy_y_map.get(rod_key, 0.0)):
                    self.lane_clear_state[rod_key] = False

            if any_back_kicking and not cocked_here:
                if not self._ball_passed_rod(ball_y, guy_y_map.get(rod_key, 0.0)):
                    self.lane_clear_state[rod_key] = True

            if self.lane_clear_state[rod_key] and not cocked_here:
                rmin, rmax = rot_lims[rod_key]
                action[rot_idx] = float(np.clip(self.clear_lane_angle, rmin, rmax))

        return action

    # ------------------------------------------------------------------ #
    # Role-specific actions
    # ------------------------------------------------------------------ #

    def _goalkeeper_action(self, rod_key, act_lin, obs, phase, stats):
        ball_x, ball_y = float(obs[0]), float(obs[1])
        ball_vx, ball_vy = float(obs[3]), float(obs[4])
        gx, gy = stats[0], stats[1]

        if phase == DEFENDING:
            pred_x = self._predict_arrival_x(ball_x, ball_y, ball_vx, ball_vy, gy)
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, pred_x, ball_y)
        else:
            base = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, ball_x, ball_y)
            center_target = 0.0
            slide_target = base * (1 - self.goalie_center_bias) + center_target * self.goalie_center_bias

        dx = abs(ball_x - gx)
        dy_front = self._dy_front(ball_y, gy)

        if self._is_fast_approach(ball_vy):
            rot_target = self.kick_sign * self.goalie_blocking_angle
            return slide_target, rot_target, False

        dy_back = -dy_front
        if dy_back > 0 and self._is_approaching_from_back(ball_vy):
            self.rod_cocked_state[rod_key] = False
            self.rod_cocked_y[rod_key] = 0.0
            ball_speed_ba = np.sqrt(ball_vx**2 + ball_vy**2)
            if dy_back <= self.back_kick_dy_max and dx <= self.kick_x_close:
                kick_angle = self.kick_strike_speed + ball_speed_ba * 0.1
                rot_target = self.kick_sign * kick_angle
                return slide_target, rot_target, True
            return slide_target, 0.0, False

        in_contact = (0.0 <= dy_front <= self.kick_y_contact_max) and (dx <= self.kick_x_contact)
        if in_contact:
            rot_target = self.kick_sign * self.goalie_clear_angle
            return slide_target, rot_target, True

        in_front_ready = (self.kick_y_front_min <= dy_front <= self.kick_y_front_max) and (dx <= self.kick_x_close)

        ball_speed = np.sqrt(ball_vx**2 + ball_vy**2)
        if (ball_speed < self.kick_back_min_speed
                and in_front_ready and dx <= self.stationary_kick_dx):
            rot_target = self.kick_sign * self.goalie_clear_angle
            return slide_target, rot_target, True

        if in_front_ready and phase == DEFENDING:
            ball_speed = np.sqrt(ball_vx**2 + ball_vy**2)
            cock_back, _ = self._kick_angles(ball_speed)
            rot_target = self.kick_sign * cock_back
            self.rod_cocked_state[rod_key] = True
            self.rod_cocked_y[rod_key] = gy
            return slide_target, rot_target, True

        rot_target = self.kick_sign * self.goalie_blocking_angle
        return slide_target, rot_target, False

    def _defense_action(self, rod_key, act_lin, obs, phase, stats):
        ball_x, ball_y = float(obs[0]), float(obs[1])
        ball_vx, ball_vy = float(obs[3]), float(obs[4])
        gx, gy = stats[0], stats[1]

        if self._is_ball_approaching(ball_vy):
            pred_x = self._predict_arrival_x(ball_x, ball_y, ball_vx, ball_vy, gy)
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, pred_x, ball_y)
        else:
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, ball_x, ball_y)

        dx = abs(ball_x - gx)
        dy_front = self._dy_front(ball_y, gy)

        if self._is_fast_approach(ball_vy):
            rot_target = self.kick_sign * self.goalie_blocking_angle
            return slide_target, rot_target, False

        ball_speed = np.sqrt(ball_vx**2 + ball_vy**2)
        pass_target = self._find_next_rod_target_x(rod_key, ball_x)
        gap_x = self._find_nearest_opponent_gap(ball_y, preferred_x=pass_target)
        aim_x = gap_x if gap_x is not None else pass_target
        return self._standard_cock_kick(rod_key, gx, gy, dx, dy_front, ball_vy, slide_target, ball_speed, ball_x, aim_x)

    def _midfield_action(self, rod_key, act_lin, obs, phase, stats):
        ball_x, ball_y = float(obs[0]), float(obs[1])
        ball_vx, ball_vy = float(obs[3]), float(obs[4])
        gx, gy = stats[0], stats[1]

        if self._is_ball_approaching(ball_vy):
            pred_x = self._predict_arrival_x(ball_x, ball_y, ball_vx, ball_vy, gy)
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, pred_x, ball_y)
        else:
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, ball_x, ball_y)

        dx = abs(ball_x - gx)
        dy_front = self._dy_front(ball_y, gy)

        if self._is_fast_approach(ball_vy):
            rot_target = self.kick_sign * self.goalie_blocking_angle
            return slide_target, rot_target, False

        ball_speed = np.sqrt(ball_vx**2 + ball_vy**2)
        pass_target = self._find_next_rod_target_x(rod_key, ball_x)
        gap_x = self._find_nearest_opponent_gap(ball_y, preferred_x=pass_target)
        aim_x = gap_x if gap_x is not None else pass_target
        return self._standard_cock_kick(rod_key, gx, gy, dx, dy_front, ball_vy, slide_target, ball_speed, ball_x, aim_x)

    def _attack_action(self, rod_key, act_lin, obs, phase, stats):
        ball_x, ball_y = float(obs[0]), float(obs[1])
        ball_vx, ball_vy = float(obs[3]), float(obs[4])
        gx, gy = stats[0], stats[1]

        if self._is_ball_approaching(ball_vy):
            pred_x = self._predict_arrival_x(ball_x, ball_y, ball_vx, ball_vy, gy)
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, pred_x, ball_y)
        else:
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, ball_x, ball_y)

        dx = abs(ball_x - gx)
        dy_front = self._dy_front(ball_y, gy)

        ball_speed = np.sqrt(ball_vx**2 + ball_vy**2)
        gap_x = self._find_nearest_opponent_gap(ball_y, preferred_x=0.0)
        aim_x = gap_x if gap_x is not None else 0.0
        return self._standard_cock_kick(rod_key, gx, gy, dx, dy_front, ball_vy, slide_target, ball_speed, ball_x, aim_x)

    # ------------------------------------------------------------------ #
    # 2-phase kick angle computation
    # ------------------------------------------------------------------ #

    def _kick_angles(self, ball_speed):
        if ball_speed < self.kick_back_min_speed:
            cock_back = self.kick_back_default
        else:
            cock_back = ball_speed * self.kick_back_speed_gain
        forward = self.kick_strike_speed
        return cock_back, forward

    # ------------------------------------------------------------------ #
    # Standard cock-and-kick
    # ------------------------------------------------------------------ #

    def _standard_cock_kick(self, rod_key, gx, gy, dx, dy_front, ball_vy, slide_target, ball_speed=0.0, ball_x=0.0, aim_x=None):
        dy_back = -dy_front
        if dy_back > 0 and self._is_approaching_from_back(ball_vy):
            self.rod_cocked_state[rod_key] = False
            self.rod_cocked_y[rod_key] = 0.0

            if dy_back <= self.back_kick_dy_max and dx <= self.kick_x_close:
                kick_angle = self.kick_strike_speed + ball_speed * 0.1
                rot_target = self.kick_sign * kick_angle
                return slide_target, rot_target, True

            return slide_target, 0.0, False

        in_front_ready = (self.kick_y_front_min <= dy_front <= self.kick_y_front_max) and (dx <= self.kick_x_close)
        in_contact = (0.0 <= dy_front <= self.kick_y_contact_max) and (dx <= self.kick_x_contact)
        approaching_front_fast = (dy_front > 0.0) and self._is_fast_approach(ball_vy)

        ball_y_approx = gy + dy_front * self.direction
        if self.rod_cocked_state[rod_key]:
            if self._ball_passed_rod(ball_y_approx, self.rod_cocked_y[rod_key]):
                self.rod_cocked_state[rod_key] = False
                self.rod_cocked_y[rod_key] = 0.0
            elif dy_front < -2.0:
                self.rod_cocked_state[rod_key] = False
                self.rod_cocked_y[rod_key] = 0.0
        else:
            if in_front_ready and (not approaching_front_fast):
                self.rod_cocked_state[rod_key] = True
                self.rod_cocked_y[rod_key] = gy

        cocked = self.rod_cocked_state[rod_key]
        cock_back, forward = self._kick_angles(ball_speed)

        if (ball_speed < self.kick_back_min_speed
                and in_front_ready and not in_contact
                and dx <= self.stationary_kick_dx):
            rot_target = self.kick_sign * forward
            if aim_x is not None:
                slide_target += (aim_x - ball_x) * 0.7
                slide_target += np.random.uniform(-0.5, 0.5)
            return slide_target, rot_target, True

        if in_contact:
            slide_target += np.random.uniform(-0.5, 0.5)

        if cocked:
            if dy_front <= self.cock_engage_dy_max:
                rot_target = self.kick_sign * cock_back
            else:
                rot_target = 0.0
            if in_contact:
                rot_target = self.kick_sign * forward
                if aim_x is not None:
                    slide_target += (aim_x - ball_x) * 0.7
                    slide_target += np.random.uniform(-0.5, 0.5)
                else:
                    slide_target += np.random.uniform(-self.kick_offset_max, self.kick_offset_max)
        else:
            rot_target = 0.0
            if approaching_front_fast and in_contact:
                rot_target = self.kick_sign * forward
                if aim_x is not None:
                    slide_target += (aim_x - ball_x) * 0.7
                    slide_target += np.random.uniform(-0.5, 0.5)
                else:
                    slide_target += np.random.uniform(-self.kick_offset_max, self.kick_offset_max)

        return slide_target, rot_target, cocked
