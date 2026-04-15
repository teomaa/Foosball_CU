"""SmartFoosballBase — shared base class for smart rule-based strategies.

Extracts common helpers from AdvancedStrategy so that multiple smart
strategies can reuse them: phase detection, ball prediction, opponent
gap analysis, direction helpers, and table geometry constants.
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


class SmartFoosballBase(FoosballStrategy):
    """Base class providing shared helpers for smart strategies.

    Subclasses must implement ``compute_action`` and ``reset``.
    """

    def __init__(self, env, team: str):
        super().__init__(env, team)

        # --- Common kick / rotation params ---
        self.kick_sign = 1.0
        self.kick_x_close = 1.5
        self.kick_x_contact = 0.6
        self.kick_y_front_min = -1.2
        self.kick_y_front_max = 9.0
        self.kick_y_contact_max = 1.25
        self.front_fast_vy_thresh = 9.0
        self.clear_lane_angle = -1.5
        self.back_kick_dy_max = 2.0
        self.stationary_kick_dx = 1.0

        # --- Phase detection thresholds ---
        self.phase_defend_y_frac = 0.35
        self.phase_attack_y_frac = 0.35
        self.phase_vel_thresh = 1.0

        # --- Table geometry ---
        self.table_x_min = -15.0
        self.table_x_max = 15.0
        self.table_y_half = 65.0

    # ------------------------------------------------------------------ #
    # Phase detection
    # ------------------------------------------------------------------ #

    def _detect_phase(self, ball_y: float, ball_vy: float) -> int:
        if self.direction == 1:
            own_y = -ball_y
            approaching_own = ball_vy < 0
        else:
            own_y = ball_y
            approaching_own = ball_vy > 0

        defend_line = self.table_y_half * self.phase_defend_y_frac
        attack_line = -self.table_y_half * self.phase_attack_y_frac

        if own_y > defend_line and approaching_own:
            return DEFENDING
        if own_y < attack_line and not approaching_own:
            return ATTACKING
        if approaching_own and abs(ball_vy) > self.phase_vel_thresh:
            return DEFENDING
        if not approaching_own and abs(ball_vy) > self.phase_vel_thresh:
            return ATTACKING
        return NEUTRAL

    # ------------------------------------------------------------------ #
    # Ball trajectory prediction
    # ------------------------------------------------------------------ #

    def _predict_arrival_x(self, ball_x, ball_y, ball_vx, ball_vy, rod_y):
        if self.direction == 1:
            dy_to_rod = rod_y - ball_y
            vy_toward_rod = ball_vy
        else:
            dy_to_rod = ball_y - rod_y
            vy_toward_rod = -ball_vy

        if vy_toward_rod <= 0.1:
            return ball_x
        if abs(ball_vy) < 0.1:
            return ball_x

        time_to_rod = abs(dy_to_rod / ball_vy)
        pred_x = ball_x + ball_vx * time_to_rod
        return self._reflect_walls(pred_x)

    def _reflect_walls(self, x):
        x_range = self.table_x_max - self.table_x_min
        x_shifted = x - self.table_x_min
        cycle = 2 * x_range
        x_mod = x_shifted % cycle if cycle > 0 else 0
        if x_mod > x_range:
            x_mod = cycle - x_mod
        return x_mod + self.table_x_min

    # ------------------------------------------------------------------ #
    # Opponent gap analysis
    # ------------------------------------------------------------------ #

    def _find_nearest_opponent_gap(self, ball_y, preferred_x=None):
        opponent = "b" if self.team == "y" else "y"

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

        ahead.sort(key=lambda t: t[0] * self.direction)
        _rod_y, guy_bids = ahead[0]
        xs = sorted([float(self.env.data.xpos[bid, 0]) for bid in guy_bids])
        if len(xs) < 2:
            return None

        gaps = []
        gaps.append(((self.table_x_min + xs[0]) / 2.0, xs[0] - self.table_x_min))
        for j in range(len(xs) - 1):
            gaps.append(((xs[j] + xs[j + 1]) / 2.0, xs[j + 1] - xs[j]))
        gaps.append(((xs[-1] + self.table_x_max) / 2.0, self.table_x_max - xs[-1]))

        if not gaps:
            return None

        if preferred_x is not None:
            min_width = 2.0
            valid = [(c, w) for c, w in gaps if w >= min_width]
            if valid:
                return min(valid, key=lambda g: abs(g[0] - preferred_x))[0]

        return max(gaps, key=lambda g: g[1])[0]

    def _find_next_rod_target_x(self, rod_key, ball_x):
        next_rod = _NEXT_ROD.get(rod_key)
        if next_rod is None:
            return None
        guy_bids = self.env._rod_guy_body_ids(self.team, next_rod)
        if not guy_bids:
            return None
        xs = [float(self.env.data.xpos[bid, 0]) for bid in guy_bids]
        if not xs:
            return None
        return min(xs, key=lambda x: abs(x - ball_x))

    # ------------------------------------------------------------------ #
    # Rod / foosman helpers
    # ------------------------------------------------------------------ #

    def _get_rod_foosman_xs(self, team, rod_key):
        """Return sorted list of X positions for all foosmen on a rod."""
        guy_bids = self.env._rod_guy_body_ids(team, rod_key)
        if not guy_bids:
            return []
        return sorted([float(self.env.data.xpos[bid, 0]) for bid in guy_bids])

    # ------------------------------------------------------------------ #
    # Direction helpers
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
        if self.direction == 1:
            return ball_vy < 0
        else:
            return ball_vy > 0

    def _is_approaching_from_back(self, ball_vy: float) -> bool:
        if self.direction == 1:
            return ball_vy > 0.0
        else:
            return ball_vy < 0.0
