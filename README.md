# Foosball_CU
Design and development of a professional foosball playing AI.
The directory is organized under 3 teams - Mechanical Assets, Simulation, and AI agents

## Ghost Opponent

The ghost opponent is a hardcoded, non-AI controller for the black team with 7 difficulty levels (0-6). It provides a curriculum of increasing challenge for training the RL agent without requiring a pre-trained model checkpoint.

### Tasks

| Task | Purpose |
|------|---------|
| `Foosball-vsghost-v0` | Training against ghost with automatic curriculum |
| `Foosball-ghostdemo-v0` | Demo/play a specific ghost level (no training) |

### Ghost Levels

| Level | Behavior |
|-------|----------|
| 0 | All rods up (same as `Foosball-1player-v0`) |
| 1 | Rods randomly up/down, 30% chance down. Holds state for ~0.25-0.5s then re-rolls. No lateral movement. |
| 2 | Same as level 1, but 60% chance down |
| 3 | All rods permanently down, no lateral movement, no kicking |
| 4 | Rods down + track ball Y-position (slow lateral movement to align nearest player figure with ball) |
| 5 | Same as level 4, plus wind-up-then-strike kick when ball approaches a rod from the front |
| 6 | Predictive tracking (leads ball based on velocity) + coordinated play (defense blocks, offense aims kicks at goal center) |

### Training with Curriculum

```bash
python IsaacLab/scripts/reinforcement_learning/sb3/train.py \
    --task Foosball-vsghost-v0 \
    --ghost_min_level 0 \
    --ghost_level_steps 100000
```

- `--ghost_min_level N` — Start at level N (skip easier levels)
- `--ghost_level_steps N` — Increase level every N env steps. Set to 0 to stay at min_level.

The current ghost level is logged to tensorboard as `ghost_level`. Opponent goals are tracked as `opponent_goal_scored_pct`.

### Demo Mode

```bash
python IsaacLab/scripts/reinforcement_learning/sb3/play.py \
    --task Foosball-ghostdemo-v0 \
    --ghost_level 4 \
    --checkpoint /path/to/model.zip
```

- `--ghost_level N` — Which level to demo (0-6)

### Tuning

Ghost behavior is controlled by constants in `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/foosball2/ghost_opponent.py`:

- `ROD_X` — X positions of black rods (may need empirical tuning from USD geometry)
- `ROD_FIGURE_OFFSETS` — Y offsets of player figures on each rod
- `PRISMATIC_KP/KD` — PD gains for lateral tracking
- `REVOLUTE_KP/KD` — PD gains for rod rotation
- `SLOW_TRACKING_EFFORT / FAST_TRACKING_EFFORT` — Effort clamps for levels 4-5 vs 6
- `KICK_WINDUP_STEPS / KICK_STRIKE_STEPS` — Timing for kick wind-up and strike phases
- `TIMER_MIN / TIMER_MAX` — Range of physics steps a rod holds its up/down state (levels 1-2)
