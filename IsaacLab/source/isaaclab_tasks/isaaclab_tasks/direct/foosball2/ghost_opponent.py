"""Ghost opponent: a hardcoded, non-AI controller for the black team with 7 difficulty levels."""

from __future__ import annotations

import math
import torch

# ---------------------------------------------------------------------------
# Constants (approximate — may need empirical tuning from USD geometry)
# ---------------------------------------------------------------------------

# X positions of the 4 black rods (Keeper, Defense, Mid, Offense)
# Positive X = black goal side, negative X = white goal side
ROD_X = torch.tensor([0.50, 0.30, 0.10, -0.15])

# Y offsets of player figures on each rod relative to prismatic=0
# Keeper: 1 figure, Defense: 2, Mid: 5, Offense: 3
ROD_FIGURE_OFFSETS: list[list[float]] = [
    [0.0],                                       # Keeper
    [-0.03, 0.03],                               # Defense
    [-0.07, -0.035, 0.0, 0.035, 0.07],          # Mid
    [-0.055, 0.0, 0.055],                        # Offense
]

# PD gains for position-based effort control
PRISMATIC_KP = 30.0
PRISMATIC_KD = 5.0
REVOLUTE_KP = 15.0
REVOLUTE_KD = 3.0

# Effort clamps for tracking
SLOW_TRACKING_EFFORT = 15.0   # levels 4-5
FAST_TRACKING_EFFORT = 40.0   # level 6

# Kick parameters
KICK_WINDUP_ANGLE = -math.pi / 2
KICK_STRIKE_EFFORT = 5.0       # matches revolute effort_limit in FOOSBALL_VS_CFG
KICK_WINDUP_STEPS = 8          # physics steps to hold wind-up
KICK_STRIKE_STEPS = 4          # physics steps for strike

# Rod up/down timer range (physics steps)
TIMER_MIN = 30
TIMER_MAX = 60

# Ball approach detection
KICK_TRIGGER_DIST = 0.08      # X distance from rod to trigger kick
KICK_TRIGGER_VEL = -0.1       # ball vx threshold (must be moving toward white goal)

# Field bounds
FIELD_Y_MAX = 0.085


class GhostOpponent:
    """Hardcoded ghost opponent for the black team with 7 difficulty levels (0-6)."""

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        self.level = 0

        # Rod X positions on device
        self.rod_x = ROD_X.to(device)

        # Pre-build figure offset tensors per rod
        self.figure_offsets = [
            torch.tensor(offsets, device=device, dtype=torch.float32)
            for offsets in ROD_FIGURE_OFFSETS
        ]

        # --- Per-env, per-rod state ---
        # Rod up/down state: 0=up, 1=down  (levels 1-2)
        self.rod_state = torch.zeros(num_envs, 4, device=device, dtype=torch.long)
        self.rod_timer = torch.randint(TIMER_MIN, TIMER_MAX + 1, (num_envs, 4), device=device)

        # Kick state machine (levels 5-6): 0=idle, 1=wind-up, 2=strike
        self.kick_phase = torch.zeros(num_envs, 4, device=device, dtype=torch.long)
        self.kick_timer = torch.zeros(num_envs, 4, device=device, dtype=torch.long)

    def set_level(self, level: int) -> None:
        self.level = level

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset ghost internal state for the given env indices."""
        self.rod_state[env_ids] = 0
        self.rod_timer[env_ids] = torch.randint(
            TIMER_MIN, TIMER_MAX + 1, (len(env_ids), 4), device=self.device
        )
        self.kick_phase[env_ids] = 0
        self.kick_timer[env_ids] = 0

    def compute_actions(
        self,
        black_pris_pos: torch.Tensor,
        black_pris_vel: torch.Tensor,
        black_rev_pos: torch.Tensor,
        black_rev_vel: torch.Tensor,
        ball_pos: torch.Tensor,
        ball_vel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (prismatic_efforts, revolute_efforts), each [num_envs, 4]."""
        if self.level == 0:
            return self._level_0(black_rev_pos, black_rev_vel, black_pris_pos)
        elif self.level <= 2:
            return self._level_1_2(black_rev_pos, black_rev_vel, black_pris_pos)
        elif self.level == 3:
            return self._level_3(black_rev_pos, black_rev_vel, black_pris_pos)
        elif self.level == 4:
            return self._level_4(
                black_pris_pos, black_pris_vel, black_rev_pos, black_rev_vel, ball_pos
            )
        elif self.level == 5:
            return self._level_5(
                black_pris_pos, black_pris_vel, black_rev_pos, black_rev_vel,
                ball_pos, ball_vel
            )
        else:
            return self._level_6(
                black_pris_pos, black_pris_vel, black_rev_pos, black_rev_vel,
                ball_pos, ball_vel
            )

    # ------------------------------------------------------------------
    # Level implementations
    # ------------------------------------------------------------------

    def _level_0(self, rev_pos, rev_vel, pris_pos):
        """All rods up, no lateral movement."""
        rev_effort = REVOLUTE_KP * (2.0 - rev_pos) - REVOLUTE_KD * rev_vel
        pris_effort = torch.zeros_like(pris_pos)
        return pris_effort, rev_effort

    def _level_1_2(self, rev_pos, rev_vel, pris_pos):
        """Random up/down per rod with timer. Level 1: 30% down, Level 2: 60% down."""
        down_prob = 0.3 if self.level == 1 else 0.6

        # Re-roll rods whose timer expired
        expired = self.rod_timer <= 0
        if expired.any():
            new_state = (torch.rand(self.num_envs, 4, device=self.device) < down_prob).long()
            self.rod_state = torch.where(expired, new_state, self.rod_state)
            new_timer = torch.randint(
                TIMER_MIN, TIMER_MAX + 1, (self.num_envs, 4), device=self.device
            )
            self.rod_timer = torch.where(expired, new_timer, self.rod_timer)

        self.rod_timer -= 1

        # Drive revolute toward 0.0 (down) or 2.0 (up) based on state
        target = torch.where(
            self.rod_state == 1,
            torch.zeros_like(rev_pos),
            2.0 * torch.ones_like(rev_pos),
        )
        rev_effort = REVOLUTE_KP * (target - rev_pos) - REVOLUTE_KD * rev_vel
        pris_effort = torch.zeros_like(pris_pos)
        return pris_effort, rev_effort

    def _level_3(self, rev_pos, rev_vel, pris_pos):
        """All rods down, no lateral movement, no kicking."""
        rev_effort = REVOLUTE_KP * (0.0 - rev_pos) - REVOLUTE_KD * rev_vel
        pris_effort = torch.zeros_like(pris_pos)
        return pris_effort, rev_effort

    def _level_4(self, pris_pos, pris_vel, rev_pos, rev_vel, ball_pos):
        """Rods down + slow ball-Y tracking."""
        rev_effort = REVOLUTE_KP * (0.0 - rev_pos) - REVOLUTE_KD * rev_vel
        pris_effort = self._compute_tracking_effort(
            ball_pos[:, 1], pris_pos, pris_vel, SLOW_TRACKING_EFFORT
        )
        return pris_effort, rev_effort

    def _level_5(self, pris_pos, pris_vel, rev_pos, rev_vel, ball_pos, ball_vel):
        """Tracking + wind-up-then-strike kick when ball approaches from front."""
        pris_effort = self._compute_tracking_effort(
            ball_pos[:, 1], pris_pos, pris_vel, SLOW_TRACKING_EFFORT
        )
        rev_effort = self._compute_kick_efforts(
            ball_pos, ball_vel, rev_pos, rev_vel, aimed=False
        )
        return pris_effort, rev_effort

    def _level_6(self, pris_pos, pris_vel, rev_pos, rev_vel, ball_pos, ball_vel):
        """Predictive tracking + coordinated kicks (offense aims at goal center)."""
        # Predict ball Y at each rod's X position
        predicted_y = self._predict_ball_y(ball_pos, ball_vel)
        pris_effort = self._compute_tracking_effort(
            predicted_y, pris_pos, pris_vel, FAST_TRACKING_EFFORT, per_rod=True
        )
        rev_effort = self._compute_kick_efforts(
            ball_pos, ball_vel, rev_pos, rev_vel, aimed=True
        )
        return pris_effort, rev_effort

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_tracking_effort(
        self,
        target_y: torch.Tensor,
        pris_pos: torch.Tensor,
        pris_vel: torch.Tensor,
        effort_limit: float,
        per_rod: bool = False,
    ) -> torch.Tensor:
        """Compute prismatic effort to align nearest figure with target_y.

        Args:
            target_y: [num_envs] if per_rod=False (same target for all rods),
                      [num_envs, 4] if per_rod=True (different target per rod).
            pris_pos: [num_envs, 4]
            pris_vel: [num_envs, 4]
            effort_limit: max absolute effort
            per_rod: whether target_y has a per-rod dimension
        """
        pris_target = torch.zeros_like(pris_pos)

        for rod_idx in range(4):
            offsets = self.figure_offsets[rod_idx]  # [n_figs]
            ty = target_y[:, rod_idx] if per_rod else target_y  # [num_envs]

            # desired_pris for each figure = target_y - offset
            candidates = ty.unsqueeze(-1) - offsets.unsqueeze(0)  # [num_envs, n_figs]

            # pick candidate closest to current position (smoothest move)
            dist = torch.abs(candidates - pris_pos[:, rod_idx : rod_idx + 1])
            best_idx = dist.argmin(dim=-1)  # [num_envs]
            pris_target[:, rod_idx] = candidates.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)

        error = pris_target - pris_pos
        effort = PRISMATIC_KP * error - PRISMATIC_KD * pris_vel
        return effort.clamp(-effort_limit, effort_limit)

    def _compute_kick_efforts(
        self,
        ball_pos: torch.Tensor,
        ball_vel: torch.Tensor,
        rev_pos: torch.Tensor,
        rev_vel: torch.Tensor,
        aimed: bool,
    ) -> torch.Tensor:
        """Kick state machine for all 4 rods. Returns revolute efforts [num_envs, 4].

        Args:
            aimed: if True (level 6), defense only kicks when ball very close,
                   offense/mid kick more aggressively.
        """
        rev_effort = torch.zeros_like(rev_pos)

        for rod_idx in range(4):
            rod_x = self.rod_x[rod_idx]

            # Kick trigger: ball near rod and moving toward white goal (vx < threshold)
            dx = torch.abs(ball_pos[:, 0] - rod_x)
            trigger_dist = KICK_TRIGGER_DIST
            if aimed:
                # Defense/Keeper (rods 0,1): only kick when ball very close
                # Offense/Mid (rods 2,3): wider trigger
                trigger_dist = 0.04 if rod_idx <= 1 else 0.10

            ball_approaching = (ball_vel[:, 0] < KICK_TRIGGER_VEL) & (dx < trigger_dist)

            idle = self.kick_phase[:, rod_idx] == 0
            windup = self.kick_phase[:, rod_idx] == 1
            strike = self.kick_phase[:, rod_idx] == 2

            # idle -> wind-up on trigger
            trigger = idle & ball_approaching
            self.kick_phase[:, rod_idx] = torch.where(
                trigger, torch.ones_like(self.kick_phase[:, rod_idx]), self.kick_phase[:, rod_idx]
            )
            self.kick_timer[:, rod_idx] = torch.where(
                trigger,
                torch.full_like(self.kick_timer[:, rod_idx], KICK_WINDUP_STEPS),
                self.kick_timer[:, rod_idx],
            )

            # Refresh masks after transitions
            windup = self.kick_phase[:, rod_idx] == 1

            # wind-up effort: drive toward KICK_WINDUP_ANGLE
            windup_effort = REVOLUTE_KP * (KICK_WINDUP_ANGLE - rev_pos[:, rod_idx]) \
                - REVOLUTE_KD * rev_vel[:, rod_idx]

            # wind-up timer
            self.kick_timer[:, rod_idx] -= windup.long()
            to_strike = windup & (self.kick_timer[:, rod_idx] <= 0)
            self.kick_phase[:, rod_idx] = torch.where(
                to_strike,
                2 * torch.ones_like(self.kick_phase[:, rod_idx]),
                self.kick_phase[:, rod_idx],
            )
            self.kick_timer[:, rod_idx] = torch.where(
                to_strike,
                torch.full_like(self.kick_timer[:, rod_idx], KICK_STRIKE_STEPS),
                self.kick_timer[:, rod_idx],
            )

            # Refresh
            strike = self.kick_phase[:, rod_idx] == 2

            # strike effort: max positive torque
            strike_effort = KICK_STRIKE_EFFORT * torch.ones_like(rev_pos[:, rod_idx])

            # strike timer
            self.kick_timer[:, rod_idx] -= strike.long()
            to_idle = strike & (self.kick_timer[:, rod_idx] <= 0)
            self.kick_phase[:, rod_idx] = torch.where(
                to_idle, torch.zeros_like(self.kick_phase[:, rod_idx]), self.kick_phase[:, rod_idx]
            )

            # Refresh
            idle = self.kick_phase[:, rod_idx] == 0

            # idle effort: hold rods down
            idle_effort = REVOLUTE_KP * (0.0 - rev_pos[:, rod_idx]) \
                - REVOLUTE_KD * rev_vel[:, rod_idx]

            rev_effort[:, rod_idx] = (
                idle_effort * idle.float()
                + windup_effort * windup.float()
                + strike_effort * strike.float()
            )

        return rev_effort

    def _predict_ball_y(self, ball_pos: torch.Tensor, ball_vel: torch.Tensor) -> torch.Tensor:
        """Predict ball Y at each rod's X position. Returns [num_envs, 4]."""
        predicted = torch.zeros(self.num_envs, 4, device=self.device)
        for rod_idx in range(4):
            rod_x = self.rod_x[rod_idx]
            dx = rod_x - ball_pos[:, 0]
            # Only predict when ball is moving toward this rod
            moving_toward = (ball_vel[:, 0] * dx) > 0
            vx_safe = ball_vel[:, 0].clone()
            vx_safe[vx_safe.abs() < 0.01] = 0.01  # avoid div by zero
            time_to_rod = torch.where(
                moving_toward,
                (dx / vx_safe).clamp(0.0, 2.0),
                torch.zeros_like(dx),
            )
            predicted[:, rod_idx] = (ball_pos[:, 1] + ball_vel[:, 1] * time_to_rod).clamp(
                -FIELD_Y_MAX, FIELD_Y_MAX
            )
        return predicted
