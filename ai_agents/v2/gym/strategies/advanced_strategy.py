"""AdvancedStrategy — smarter rule-based agent implementing PROPOSED_RULES.md.

Key improvements over BasicStrategy:
  - Game-phase detection (DEFENDING / NEUTRAL / ATTACKING)
  - Role-specific behavior for each rod (goalkeeper, defense, midfield, attack)
  - Ball trajectory prediction with wall-bounce reflection
  - Opponent gap analysis for aimed shots
  - Deliberate pass targeting toward next rod's foosman
"""

import numpy as np

from ai_agents.v2.gym.strategies.base_strategy import FoosballStrategy

ROD_KEYS = ["goal", "def", "mid", "attack"]

# Forward rod mapping for pass targeting
_NEXT_ROD = {"goal": "def", "def": "mid", "mid": "attack"}

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
        self.kick_back_speed_gain = 0.02
        # Default cock-back angle when ball is slow/stationary (rad).
        # Smaller = slower cocking, and larger forward swing (faster kick).
        self.kick_back_default = 0.5
        # Ball speed threshold below which kick_back_default is used.
        self.kick_back_min_speed = 0.5
        # Static forward kick speed (rad). Fixed high value, independent of cock-back.
        # Ensures maximum forward rotation on every kick.
        self.kick_strike_speed = 10.0
        # X-distance (table units) to start cocking. Higher = starts cocking from further away, more anticipation but more false triggers.
        self.kick_x_close = 1.5
        # X-distance for contact kick. Lower = requires tighter alignment to fire, more accurate but easier to miss.
        self.kick_x_contact = 0.6
        # Min Y-distance (ball ahead of foosman) to begin cocking. More negative = starts cocking when ball is slightly behind.
        self.kick_y_front_min = -1.2
        # Max Y-distance for rod to be "aware" of ball (ready zone). Must be large enough
        # to cover the ball's starting position relative to the nearest rod (~7.5 units).
        self.kick_y_front_max = 9.0
        # Max Y-distance to actually engage the cock-back rotation. Ball must be this close
        # before the rod winds up. Further away = hold neutral (0) instead of cocking.
        self.cock_engage_dy_max = 3.0
        # Max Y-distance for contact kick. Higher = fires at ball that's further in front, more forgiving but less precise.
        self.kick_y_contact_max = 1.25
        # Random slide offset on kick (table units). Higher = wider angle shot variety, less predictable but less accurate.
        self.kick_offset_max = 5.0
        # Lane-clearing rotation (rad). More negative = lifts foosmen higher out of ball's path, but slower to recover.
        self.clear_lane_angle = -1.5
        # Ball vy threshold for "fast approach" (table units/sec). Lower = more balls treated as fast (hold block instead of cocking).
        self.front_fast_vy_thresh = 9.0
        # Max Y-distance behind rod to trigger back-approach kick.
        # Only react when ball is very close behind (small value = tighter window).
        self.back_kick_dy_max = 2.0

        # Max X-distance to consider ball near enough for a stationary kick.
        self.stationary_kick_dx = 1.0

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
        self.goalie_clear_angle = 5.0

        # --- Table geometry (hardcoded, can refine via MuJoCo query) ---
        self.table_x_min = -15.0
        self.table_x_max = 15.0
        self.table_y_half = 65.0  # TABLE_MAX_Y_DIM

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

        # Back-approach: ball behind goalkeeper and moving forward
        dy_back = -dy_front
        if dy_back > 0 and self._is_approaching_from_back(ball_vy):
            self.rod_cocked_state[rod_key] = False
            self.rod_cocked_y[rod_key] = 0.0
            # Only kick when ball is very close behind
            ball_speed_ba = np.sqrt(ball_vx**2 + ball_vy**2)
            if dy_back <= self.back_kick_dy_max and dx <= self.kick_x_close:
                kick_angle = self.kick_strike_speed + ball_speed_ba * 0.1
                rot_target = self.kick_sign * kick_angle
                return slide_target, rot_target, True
            # Not close: stay neutral
            return slide_target, 0.0, False

        # Ball very close and slow => clearing kick
        in_contact = (0.0 <= dy_front <= self.kick_y_contact_max) and (dx <= self.kick_x_contact)
        if in_contact:
            rot_target = self.kick_sign * self.goalie_clear_angle
            return slide_target, rot_target, True

        # Close enough to cock
        in_front_ready = (self.kick_y_front_min <= dy_front <= self.kick_y_front_max) and (dx <= self.kick_x_close)

        # Stationary ball near goalie but outside tight contact zone: clear it
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

        # Standard cock-and-kick — aim through nearest opponent gap
        ball_speed = np.sqrt(ball_vx**2 + ball_vy**2)
        pass_target = self._find_next_rod_target_x(rod_key, ball_x)
        gap_x = self._find_nearest_opponent_gap(ball_y, preferred_x=pass_target)
        aim_x = gap_x if gap_x is not None else pass_target
        return self._standard_cock_kick(rod_key, gx, gy, dx, dy_front, ball_vy, slide_target, ball_speed, ball_x, aim_x)

    def _midfield_action(self, rod_key, act_lin, obs, phase, stats):
        """Midfield: block when defending, pass forward when attacking/neutral."""
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

        # Standard cock-and-kick — aim through nearest opponent gap toward attack rod
        ball_speed = np.sqrt(ball_vx**2 + ball_vy**2)
        pass_target = self._find_next_rod_target_x(rod_key, ball_x)
        gap_x = self._find_nearest_opponent_gap(ball_y, preferred_x=pass_target)
        aim_x = gap_x if gap_x is not None else pass_target
        return self._standard_cock_kick(rod_key, gx, gy, dx, dy_front, ball_vy, slide_target, ball_speed, ball_x, aim_x)

    def _attack_action(self, rod_key, act_lin, obs, phase, stats):
        """Attack: aimed shot at opponent gaps."""
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

        # Standard cock-and-kick aimed through nearest opponent gap (or goal center)
        ball_speed = np.sqrt(ball_vx**2 + ball_vy**2)
        gap_x = self._find_nearest_opponent_gap(ball_y, preferred_x=0.0)
        aim_x = gap_x if gap_x is not None else 0.0
        return self._standard_cock_kick(rod_key, gx, gy, dx, dy_front, ball_vy, slide_target, ball_speed, ball_x, aim_x)

    # ------------------------------------------------------------------ #
    # 2-phase kick angle computation
    # ------------------------------------------------------------------ #

    def _kick_angles(self, ball_speed):
        """2-phase kick: slow variable cock-back (phase 1) + fast static forward (phase 2).
        Returns (cock_back_angle, forward_angle).
        Forward kick is a fixed high value, independent of cock-back."""
        if ball_speed < self.kick_back_min_speed:
            cock_back = self.kick_back_default
        else:
            cock_back = ball_speed * self.kick_back_speed_gain
        forward = self.kick_strike_speed
        return cock_back, forward

    # ------------------------------------------------------------------ #
    # Standard cock-and-kick (fallback, same as BasicStrategy)
    # ------------------------------------------------------------------ #

    def _standard_cock_kick(self, rod_key, gx, gy, dx, dy_front, ball_vy, slide_target, ball_speed=0.0, ball_x=0.0, aim_x=None):
        """Standard cock-and-kick logic, returns (slide, rot, cocked)."""

        # --- Back-approach handling: ball behind rod and moving forward ---
        dy_back = -dy_front
        if dy_back > 0 and self._is_approaching_from_back(ball_vy):
            self.rod_cocked_state[rod_key] = False
            self.rod_cocked_y[rod_key] = 0.0

            # Only kick when ball is very close behind and X-aligned
            if dy_back <= self.back_kick_dy_max and dx <= self.kick_x_close:
                # Kick strength proportional to ball approach speed
                kick_angle = self.kick_strike_speed + ball_speed * 0.1
                rot_target = self.kick_sign * kick_angle
                return slide_target, rot_target, True

            # Ball behind but not close: stay neutral (don't lift)
            return slide_target, 0.0, False

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

        # Stationary ball in range but outside tight contact zone: kick directly.
        # The rotation arc is large enough to sweep through the ball.
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
            # Only apply cock-back rotation when ball is close; hold neutral otherwise
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

    def _find_nearest_opponent_gap(self, ball_y, preferred_x=None):
        """Find a gap in the nearest opponent rod ahead of the ball.

        Checks the first opponent rod the ball will encounter in the attack
        direction, not just goal/def.  If *preferred_x* is given (e.g. the
        pass-target or goal center), returns the gap whose centre is closest
        to that value (among gaps wider than 2 units).  Otherwise returns the
        widest gap.

        Returns gap center X, or None if no opponent rod is ahead.
        """
        opponent = "b" if self.team == "y" else "y"

        # Gather opponent rods that are ahead of ball_y in attack direction
        ahead = []
        for rod_key in ROD_KEYS:
            guy_bids = self.env._rod_guy_body_ids(opponent, rod_key)
            if not guy_bids:
                continue
            rod_y = float(self.env.data.xpos[guy_bids[0], 1])
            if self.direction == 1 and rod_y > ball_y:
                ahead.append((rod_y, guy_bids))
            elif self.direction == -1 and rod_y < ball_y:
                ahead.append((rod_y, guy_bids))

        if not ahead:
            return None

        # Sort nearest-first
        ahead.sort(key=lambda t: t[0] * self.direction)

        # Check nearest opponent rod
        _rod_y, guy_bids = ahead[0]
        xs = sorted([float(self.env.data.xpos[bid, 0]) for bid in guy_bids])
        if len(xs) < 2:
            return None

        # Collect all gaps: (center_x, width)
        gaps = []
        # left wall → leftmost foosman
        gaps.append(((self.table_x_min + xs[0]) / 2.0, xs[0] - self.table_x_min))
        # between adjacent foosmen
        for j in range(len(xs) - 1):
            gaps.append(((xs[j] + xs[j + 1]) / 2.0, xs[j + 1] - xs[j]))
        # rightmost foosman → right wall
        gaps.append(((xs[-1] + self.table_x_max) / 2.0, self.table_x_max - xs[-1]))

        if not gaps:
            return None

        # If a preferred direction is given, pick the nearest sufficiently-wide gap
        if preferred_x is not None:
            min_width = 2.0
            valid = [(c, w) for c, w in gaps if w >= min_width]
            if valid:
                return min(valid, key=lambda g: abs(g[0] - preferred_x))[0]

        # Fallback: widest gap
        return max(gaps, key=lambda g: g[1])[0]

    def _find_next_rod_target_x(self, rod_key, ball_x):
        """Find X of the nearest foosman on the next forward friendly rod.

        Used for deliberate pass targeting — aim passes at a specific
        receiving foosman rather than kicking blindly forward.
        Returns target X or None if no next rod or no foosmen found.
        """
        next_rod = _NEXT_ROD.get(rod_key)
        if next_rod is None:
            return None
        guy_bids = self.env._rod_guy_body_ids(self.team, next_rod)
        if not guy_bids:
            return None
        xs = [float(self.env.data.xpos[bid, 0]) for bid in guy_bids]
        if not xs:
            return None
        # Pick the foosman closest to the ball for best reception
        return min(xs, key=lambda x: abs(x - ball_x))

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
