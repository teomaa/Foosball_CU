"""AdvancedStrategy — smarter rule-based agent implementing PROPOSED_RULES.md.

Key improvements over BasicStrategy:
  - Game-phase detection (DEFENDING / NEUTRAL / ATTACKING)
  - Role-specific behavior for each rod (goalkeeper, defense, midfield, attack)
  - Ball trajectory prediction with wall-bounce reflection
  - Opponent gap analysis for aimed shots
  - Trap-then-act (pin ball, hold, then kick/pass)
"""

import numpy as np

from ai_agents.v2.gym.strategies.base_strategy import FoosballStrategy

ROD_KEYS = ["goal", "def", "mid", "attack"]

# Game phases
DEFENDING = 0
NEUTRAL = 1
ATTACKING = 2


class AdvancedStrategy(FoosballStrategy):

    def __init__(self, env, team: str):
        super().__init__(env, team)

        # --- Kick / rotation params ---
        # +1 or -1; flips kick rotation direction. Change to -1.0 if foosmen rotate the wrong way.
        self.kick_sign = 1.0
        # --- 2-phase kick params ---
        # Phase 1 (cock-back → center): angle = ball_speed * gain (variable).
        # Phase 2 (center → forward): angle = kick_strike_speed (static).
        # Total forward target on contact = cock_back + kick_strike_speed.
        self.kick_back_speed_gain = 0.05
        # Default cock-back angle when ball is slow/stationary (rad).
        self.kick_back_default = 1.6
        # Ball speed threshold below which kick_back_default is used.
        self.kick_back_min_speed = 0.5
        # Static forward rotation past center (phase 2, rad).
        # Higher = harder kick from center onward.
        self.kick_strike_speed = 4.3
        # X-distance (table units) to start cocking. Higher = starts cocking from further away, more anticipation but more false triggers.
        self.kick_x_close = 2.5
        # X-distance for contact kick. Lower = requires tighter alignment to fire, more accurate but easier to miss.
        self.kick_x_contact = 0.6
        # Min Y-distance (ball ahead of foosman) to begin cocking. More negative = starts cocking when ball is slightly behind.
        self.kick_y_front_min = -1.2
        # Max Y-distance to begin cocking. Higher = cocks even when ball is far ahead, wider ready window.
        self.kick_y_front_max = 9.0
        # Max Y-distance for contact kick. Higher = fires at ball that's further in front, more forgiving but less precise.
        self.kick_y_contact_max = 1.25
        # Random slide offset on kick (table units). Higher = wider angle shot variety, less predictable but less accurate.
        self.kick_offset_max = 5.0
        # Lane-clearing rotation (rad). More negative = lifts foosmen higher out of ball's path, but slower to recover.
        self.clear_lane_angle = -2.0
        # Ball vy threshold for "fast approach" (table units/sec). Lower = more balls treated as fast (hold block instead of cocking).
        self.front_fast_vy_thresh = 9.0
        # Min Y-distance behind rod to trigger back-approach lift/kick.
        self.back_approach_dy_min = 0.5

        # --- Trap params ---
        # Steps to hold pin before releasing kick/pass. Higher = longer possession, more time to aim but slower play.
        self.trap_hold_steps = 3
        # Downward pin rotation (rad). More negative = presses ball harder against table, firmer trap but harder to release cleanly.
        self.trap_pin_angle = -0.4
        # Max X-distance to consider ball "possessed". Higher = attempts traps from further away, more aggressive but more failed traps.
        self.trap_dx_thresh = 1.0
        # Max Y-distance to consider ball "possessed". Higher = larger possession zone in front of foosman.
        self.trap_dy_thresh = 1.5
        # Max ball speed to attempt a trap. Higher = tries to trap faster balls, riskier but gains possession more often.
        self.trap_vel_thresh = 3.0

        # --- Phase detection thresholds ---
        # Fraction of half-field for defending zone. Higher = wider defensive zone, spends more time in DEFENDING phase.
        self.phase_defend_y_frac = 0.35
        # Fraction of half-field for attacking zone. Higher = wider attack zone, spends more time in ATTACKING phase.
        self.phase_attack_y_frac = 0.35
        # Min |vy| to assign directional phase. Lower = classifies slower balls as attacking/defending instead of NEUTRAL.
        self.phase_vel_thresh = 1.0

        # --- Goalkeeper params ---
        # Default blocking posture (rad). Higher = wider angle coverage but larger gaps between foosmen.
        self.goalie_blocking_angle = 0.4
        # Slide bias toward center when ball is far (0-1). Higher = goalie stays more centered, better coverage but less reactive to wide shots.
        self.goalie_center_bias = 0.6
        # Clearing kick angle (rad). Higher = more powerful clearance, ball travels further but less control.
        self.goalie_clear_angle = 3.90

        # --- Pass params ---
        # Kick angle for passing (rad). Lower = softer pass with more control, higher = harder pass that travels further.
        self.pass_kick_angle = 2.5

        # --- Table geometry (hardcoded, can refine via MuJoCo query) ---
        self.table_x_min = -15.0
        self.table_x_max = 15.0
        self.table_y_half = 65.0  # TABLE_MAX_Y_DIM

        # --- Persistent state ---
        self.rod_cocked_state = {k: False for k in ROD_KEYS}
        self.rod_cocked_y = {k: 0.0 for k in ROD_KEYS}
        self.lane_clear_state = {k: False for k in ROD_KEYS}
        self.trap_state = {k: False for k in ROD_KEYS}
        self.trap_counter = {k: 0 for k in ROD_KEYS}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def reset(self):
        for k in ROD_KEYS:
            self.rod_cocked_state[k] = False
            self.rod_cocked_y[k] = 0.0
            self.lane_clear_state[k] = False
            self.trap_state[k] = False
            self.trap_counter[k] = 0

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
    # Phase detection
    # ------------------------------------------------------------------ #

    def _detect_phase(self, ball_y: float, ball_vy: float) -> int:
        """Determine game phase from ball position and velocity."""
        # Normalize ball_y to own-half coordinates:
        # own_y > 0 means ball is in our half, < 0 means opponent half
        if self.direction == 1:  # yellow attacks +Y, defends -Y
            own_y = -ball_y  # positive = our half (near -Y goal)
            approaching_own = ball_vy < 0  # moving toward -Y = our goal
        else:  # black attacks -Y, defends +Y
            own_y = ball_y   # positive = our half (near +Y goal)
            approaching_own = ball_vy > 0  # moving toward +Y = our goal

        defend_line = self.table_y_half * self.phase_defend_y_frac
        attack_line = -self.table_y_half * self.phase_attack_y_frac

        if own_y > defend_line and approaching_own:
            return DEFENDING
        if own_y < attack_line and not approaching_own:
            return ATTACKING
        # Ball approaching us with decent speed => defending
        if approaching_own and abs(ball_vy) > self.phase_vel_thresh:
            return DEFENDING
        # Ball moving away with decent speed => attacking
        if not approaching_own and abs(ball_vy) > self.phase_vel_thresh:
            return ATTACKING
        return NEUTRAL

    # ------------------------------------------------------------------ #
    # Role-specific actions
    # ------------------------------------------------------------------ #

    def _goalkeeper_action(self, rod_key, act_lin, obs, phase, stats):
        """Goalkeeper: block shots, center bias, clearing kick only."""
        ball_x, ball_y = float(obs[0]), float(obs[1])
        ball_vx, ball_vy = float(obs[3]), float(obs[4])
        gx, gy = stats[0], stats[1]

        # Slide: predicted interception when defending, else center bias
        if phase == DEFENDING:
            pred_x = self._predict_arrival_x(ball_x, ball_y, ball_vx, ball_vy, gy)
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, pred_x, ball_y)
        else:
            # Bias toward center
            base = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, ball_x, ball_y)
            center_target = 0.0
            slide_target = base * (1 - self.goalie_center_bias) + center_target * self.goalie_center_bias

        # Rotation: blocking angle by default
        dx = abs(ball_x - gx)
        dy_front = self._dy_front(ball_y, gy)

        # Fast approach => hold blocking angle, never cock
        if self._is_fast_approach(ball_vy):
            rot_target = self.kick_sign * self.goalie_blocking_angle
            return slide_target, rot_target, False

        # Back-approach: ball behind goalkeeper and moving further behind — lift out of the way
        dy_back = -dy_front
        if dy_back > self.back_approach_dy_min and self._is_approaching_from_back(ball_vy):
            self.rod_cocked_state[rod_key] = False
            self.rod_cocked_y[rod_key] = 0.0
            rot_target = self.clear_lane_angle
            return slide_target, rot_target, False

        # Ball very close and slow => clearing kick
        in_contact = (0.0 <= dy_front <= self.kick_y_contact_max) and (dx <= self.kick_x_contact)
        if in_contact:
            rot_target = self.kick_sign * self.goalie_clear_angle
            return slide_target, rot_target, True

        # Close enough to cock
        in_front_ready = (self.kick_y_front_min <= dy_front <= self.kick_y_front_max) and (dx <= self.kick_x_close)
        if in_front_ready and phase == DEFENDING:
            ball_speed = np.sqrt(ball_vx**2 + ball_vy**2)
            cock_back, _ = self._kick_angles(ball_speed)
            rot_target = self.kick_sign * cock_back
            self.rod_cocked_state[rod_key] = True
            self.rod_cocked_y[rod_key] = gy
            return slide_target, rot_target, True

        # Default blocking posture
        rot_target = self.kick_sign * self.goalie_blocking_angle
        return slide_target, rot_target, False

    def _defense_action(self, rod_key, act_lin, obs, phase, stats):
        """Defense: intercept + controlled pass to mid."""
        ball_x, ball_y = float(obs[0]), float(obs[1])
        ball_vx, ball_vy = float(obs[3]), float(obs[4])
        gx, gy = stats[0], stats[1]

        # Slide: predict arrival when ball approaching, else track
        if self._is_ball_approaching(ball_vy):
            pred_x = self._predict_arrival_x(ball_x, ball_y, ball_vx, ball_vy, gy)
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, pred_x, ball_y)
        else:
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, ball_x, ball_y)

        dx = abs(ball_x - gx)
        dy_front = self._dy_front(ball_y, gy)

        # Fast approach => hold blocking angle
        if self._is_fast_approach(ball_vy):
            rot_target = self.kick_sign * self.goalie_blocking_angle
            return slide_target, rot_target, False

        # Trap-then-pass logic
        s, r, cocked = self._trap_and_pass_rotation(rod_key, gx, gy, dx, dy_front, ball_vy, slide_target)
        if s is not None:
            return s, r, cocked

        # Standard cock-and-kick fallback
        ball_speed = np.sqrt(ball_vx**2 + ball_vy**2)
        return self._standard_cock_kick(rod_key, gx, gy, dx, dy_front, ball_vy, slide_target, ball_speed)

    def _midfield_action(self, rod_key, act_lin, obs, phase, stats):
        """Midfield: block when defending, trap + pass when attacking/neutral."""
        ball_x, ball_y = float(obs[0]), float(obs[1])
        ball_vx, ball_vy = float(obs[3]), float(obs[4])
        gx, gy = stats[0], stats[1]

        # Slide: predict or track
        if self._is_ball_approaching(ball_vy):
            pred_x = self._predict_arrival_x(ball_x, ball_y, ball_vx, ball_vy, gy)
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, pred_x, ball_y)
        else:
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, ball_x, ball_y)

        dx = abs(ball_x - gx)
        dy_front = self._dy_front(ball_y, gy)

        # Fast approach => block
        if self._is_fast_approach(ball_vy):
            rot_target = self.kick_sign * self.goalie_blocking_angle
            return slide_target, rot_target, False

        # Defending phase => standard kick (clear quickly)
        ball_speed = np.sqrt(ball_vx**2 + ball_vy**2)
        if phase == DEFENDING:
            return self._standard_cock_kick(rod_key, gx, gy, dx, dy_front, ball_vy, slide_target, ball_speed)

        # Otherwise trap-then-pass
        s, r, cocked = self._trap_and_pass_rotation(rod_key, gx, gy, dx, dy_front, ball_vy, slide_target)
        if s is not None:
            return s, r, cocked

        return self._standard_cock_kick(rod_key, gx, gy, dx, dy_front, ball_vy, slide_target, ball_speed)

    def _attack_action(self, rod_key, act_lin, obs, phase, stats):
        """Attack: trap + aimed shot at opponent gaps."""
        ball_x, ball_y = float(obs[0]), float(obs[1])
        ball_vx, ball_vy = float(obs[3]), float(obs[4])
        gx, gy = stats[0], stats[1]

        # Slide: predict or track
        if self._is_ball_approaching(ball_vy):
            pred_x = self._predict_arrival_x(ball_x, ball_y, ball_vx, ball_vy, gy)
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, pred_x, ball_y)
        else:
            slide_target = self.env._slide_target_for_rod_toward_ball(
                self.team, rod_key, act_lin, ball_x, ball_y)

        dx = abs(ball_x - gx)
        dy_front = self._dy_front(ball_y, gy)

        # Try trap-then-shoot with gap aiming
        ball_speed = np.sqrt(ball_vx**2 + ball_vy**2)
        s, r, cocked = self._trap_and_shoot_rotation(rod_key, gx, gy, dx, dy_front, ball_x, ball_vy, slide_target, ball_speed)
        if s is not None:
            return s, r, cocked

        # Fallback: standard cock-and-kick
        return self._standard_cock_kick(rod_key, gx, gy, dx, dy_front, ball_vy, slide_target, ball_speed)

    # ------------------------------------------------------------------ #
    # Trap-then-act
    # ------------------------------------------------------------------ #

    def _trap_and_pass_rotation(self, rod_key, gx, gy, dx, dy_front, ball_vy, slide_target):
        """Trap ball, hold, then pass forward with moderate kick.
        Returns (slide, rot, cocked) or (None, None, None) if not applicable."""
        ball_speed = abs(ball_vy)

        # Check if already trapping
        if self.trap_state[rod_key]:
            self.trap_counter[rod_key] += 1
            if self.trap_counter[rod_key] >= self.trap_hold_steps:
                # Release: pass forward
                self.trap_state[rod_key] = False
                self.trap_counter[rod_key] = 0
                rot_target = self.kick_sign * self.pass_kick_angle
                return slide_target, rot_target, True
            # Still holding
            rot_target = self.kick_sign * self.trap_pin_angle
            return slide_target, rot_target, False

        # Start trap if ball is close, slow, and in front
        if (dx <= self.trap_dx_thresh and
                0.0 <= dy_front <= self.trap_dy_thresh and
                ball_speed < self.trap_vel_thresh):
            self.trap_state[rod_key] = True
            self.trap_counter[rod_key] = 0
            rot_target = self.kick_sign * self.trap_pin_angle
            return slide_target, rot_target, False

        return None, None, None

    def _trap_and_shoot_rotation(self, rod_key, gx, gy, dx, dy_front, ball_x, ball_vy, slide_target, ball_speed=0.0):
        """Trap ball, hold, then shoot at widest opponent gap.
        Returns (slide, rot, cocked) or (None, None, None) if not applicable."""

        # Check if already trapping
        if self.trap_state[rod_key]:
            self.trap_counter[rod_key] += 1
            if self.trap_counter[rod_key] >= self.trap_hold_steps:
                # Release: aimed shot
                self.trap_state[rod_key] = False
                self.trap_counter[rod_key] = 0

                gap_x = self._find_opponent_gap()
                if gap_x is not None:
                    slide_offset = (gap_x - ball_x) * 0.3  # partial correction
                    slide_target += slide_offset
                    slide_target += np.random.uniform(-0.5, 0.5)
                else:
                    slide_target += np.random.uniform(-self.kick_offset_max, self.kick_offset_max)

                _, forward = self._kick_angles(ball_speed)
                rot_target = self.kick_sign * forward
                return slide_target, rot_target, True
            # Still holding
            rot_target = self.kick_sign * self.trap_pin_angle
            return slide_target, rot_target, False

        # Start trap if ball is close, slow, and in front
        if (dx <= self.trap_dx_thresh and
                0.0 <= dy_front <= self.trap_dy_thresh and
                ball_speed < self.trap_vel_thresh):
            self.trap_state[rod_key] = True
            self.trap_counter[rod_key] = 0
            rot_target = self.kick_sign * self.trap_pin_angle
            return slide_target, rot_target, False

        return None, None, None

    # ------------------------------------------------------------------ #
    # 2-phase kick angle computation
    # ------------------------------------------------------------------ #

    def _kick_angles(self, ball_speed):
        """2-phase kick: variable cock-back (phase 1) + static forward (phase 2).
        Returns (cock_back_angle, forward_angle)."""
        if ball_speed < self.kick_back_min_speed:
            cock_back = self.kick_back_default
        else:
            cock_back = ball_speed * self.kick_back_speed_gain
        forward = cock_back + self.kick_strike_speed
        return cock_back, forward

    # ------------------------------------------------------------------ #
    # Standard cock-and-kick (fallback, same as BasicStrategy)
    # ------------------------------------------------------------------ #

    def _standard_cock_kick(self, rod_key, gx, gy, dx, dy_front, ball_vy, slide_target, ball_speed=0.0):
        """Standard cock-and-kick logic, returns (slide, rot, cocked)."""

        # --- Back-approach handling: ball is behind this rod and moving further behind ---
        dy_back = -dy_front
        if dy_back > self.back_approach_dy_min and self._is_approaching_from_back(ball_vy):
            # Clear any stale cock state so we don't swing into the ball's path
            self.rod_cocked_state[rod_key] = False
            self.rod_cocked_y[rod_key] = 0.0

            # Opportunistic back-kick: ball is close behind, slow, and aligned
            if (dy_back <= self.kick_y_contact_max and
                    dx <= self.kick_x_contact and
                    ball_speed < self.trap_vel_thresh):
                _, forward = self._kick_angles(ball_speed)
                rot_target = self.kick_sign * forward
                return slide_target, rot_target, True

            # Otherwise just lift foosmen out of the way
            rot_target = self.clear_lane_angle
            return slide_target, rot_target, False

        in_front_ready = (self.kick_y_front_min <= dy_front <= self.kick_y_front_max) and (dx <= self.kick_x_close)
        in_contact = (0.0 <= dy_front <= self.kick_y_contact_max) and (dx <= self.kick_x_contact)
        approaching_front_fast = (dy_front > 0.0) and self._is_fast_approach(ball_vy)

        # Persistent cock state
        ball_y_approx = gy + dy_front * self.direction  # reconstruct ball_y
        if self.rod_cocked_state[rod_key]:
            if self._ball_passed_rod(ball_y_approx, self.rod_cocked_y[rod_key]):
                self.rod_cocked_state[rod_key] = False
                self.rod_cocked_y[rod_key] = 0.0
            elif dy_front < -2.0:
                # Safety net: ball significantly behind rod, force-clear stale cock state
                self.rod_cocked_state[rod_key] = False
                self.rod_cocked_y[rod_key] = 0.0
        else:
            if in_front_ready and (not approaching_front_fast):
                self.rod_cocked_state[rod_key] = True
                self.rod_cocked_y[rod_key] = gy

        cocked = self.rod_cocked_state[rod_key]
        cock_back, forward = self._kick_angles(ball_speed)

        if in_contact:
            slide_target += np.random.uniform(-0.5, 0.5)

        if cocked:
            rot_target = self.kick_sign * cock_back
            if in_contact:
                rot_target = self.kick_sign * forward
                slide_target += np.random.uniform(-self.kick_offset_max, self.kick_offset_max)
        else:
            rot_target = 0.0
            if approaching_front_fast and in_contact:
                rot_target = self.kick_sign * forward
                slide_target += np.random.uniform(-self.kick_offset_max, self.kick_offset_max)

        return slide_target, rot_target, cocked

    # ------------------------------------------------------------------ #
    # Ball trajectory prediction
    # ------------------------------------------------------------------ #

    def _predict_arrival_x(self, ball_x, ball_y, ball_vx, ball_vy, rod_y):
        """Predict where the ball will be in X when it reaches rod_y.

        Uses linear extrapolation with wall-bounce reflection.
        """
        # Compute dy from ball to rod in attack direction
        if self.direction == 1:  # yellow
            dy_to_rod = rod_y - ball_y
            vy_toward_rod = ball_vy  # positive = toward +Y rods
        else:  # black
            dy_to_rod = ball_y - rod_y
            vy_toward_rod = -ball_vy  # black rods at higher Y, ball moving -Y

        # Only predict if ball is moving toward the rod
        if vy_toward_rod <= 0.1:
            return ball_x  # fallback to current position

        if abs(ball_vy) < 0.1:
            return ball_x

        time_to_rod = abs(dy_to_rod / ball_vy)
        pred_x = ball_x + ball_vx * time_to_rod

        # Wall-bounce reflection
        pred_x = self._reflect_walls(pred_x)

        return pred_x

    def _reflect_walls(self, x):
        """Reflect predicted X off table walls."""
        x_range = self.table_x_max - self.table_x_min
        # Normalize to [0, range]
        x_shifted = x - self.table_x_min
        # Use modular reflection
        cycle = 2 * x_range
        x_mod = x_shifted % cycle if cycle > 0 else 0
        if x_mod > x_range:
            x_mod = cycle - x_mod
        return x_mod + self.table_x_min

    # ------------------------------------------------------------------ #
    # Opponent gap analysis
    # ------------------------------------------------------------------ #

    def _find_opponent_gap(self):
        """Find the X coordinate of the widest gap in the opponent's nearest defensive rod.

        Returns gap center X or None if no gaps found.
        """
        opponent = "b" if self.team == "y" else "y"

        # Check opponent goal and def rods (the ones closest to their goal)
        for opp_rod in ["goal", "def"]:
            guy_bids = self.env._rod_guy_body_ids(opponent, opp_rod)
            if not guy_bids:
                continue

            # Get foosman X positions
            xs = sorted([float(self.env.data.xpos[bid, 0]) for bid in guy_bids])
            if len(xs) < 2:
                continue

            # Check gaps between adjacent foosmen and between walls and outermost
            best_gap_center = None
            best_gap_width = 0.0

            # Gap between left wall and leftmost foosman
            gap_w = xs[0] - self.table_x_min
            if gap_w > best_gap_width:
                best_gap_width = gap_w
                best_gap_center = (self.table_x_min + xs[0]) / 2.0

            # Gaps between adjacent foosmen
            for j in range(len(xs) - 1):
                gap_w = xs[j + 1] - xs[j]
                if gap_w > best_gap_width:
                    best_gap_width = gap_w
                    best_gap_center = (xs[j] + xs[j + 1]) / 2.0

            # Gap between rightmost foosman and right wall
            gap_w = self.table_x_max - xs[-1]
            if gap_w > best_gap_width:
                best_gap_width = gap_w
                best_gap_center = (xs[-1] + self.table_x_max) / 2.0

            if best_gap_center is not None:
                return best_gap_center

        return None

    # ------------------------------------------------------------------ #
    # Direction helpers (same as BasicStrategy)
    # ------------------------------------------------------------------ #

    def _dy_front(self, ball_y: float, guy_y: float) -> float:
        if self.direction == 1:
            return ball_y - guy_y
        else:
            return guy_y - ball_y

    def _dy_back(self, ball_y: float, guy_y: float) -> float:
        if self.direction == 1:
            return guy_y - ball_y
        else:
            return ball_y - guy_y

    def _ball_passed_rod(self, ball_y: float, rod_y: float) -> bool:
        if self.direction == 1:
            return ball_y > rod_y
        else:
            return ball_y < rod_y

    def _is_fast_approach(self, ball_vy: float) -> bool:
        if self.direction == 1:
            return ball_vy < -self.front_fast_vy_thresh
        else:
            return ball_vy > self.front_fast_vy_thresh

    def _is_ball_approaching(self, ball_vy: float) -> bool:
        """Is ball moving toward our goal (i.e., we should be defending/intercepting)?"""
        if self.direction == 1:  # yellow defends -Y
            return ball_vy < 0
        else:  # black defends +Y
            return ball_vy > 0

    def _is_approaching_from_back(self, ball_vy: float) -> bool:
        if self.direction == 1:
            return ball_vy > 0.0
        else:
            return ball_vy < 0.0
