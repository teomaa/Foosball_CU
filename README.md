# Foosball_CU

Design and development of a professional foosball playing AI using reinforcement learning on NVIDIA IsaacLab (Isaac Sim).

## Quick Start

Training and inference use two scripts in `IsaacLab/scripts/reinforcement_learning/sb3/`:

```bash
# Train PPO agent against static opponent (black rods locked up)
python IsaacLab/scripts/reinforcement_learning/sb3/train.py --task Foosball-1player-v0

# Watch a trained agent play
python IsaacLab/scripts/reinforcement_learning/sb3/play.py --task Foosball-1player-v0 --checkpoint /path/to/model.zip
```

## Environments

| Task ID | Description |
|---------|-------------|
| `Foosball-1player-v0` | Single-player training. Black team rods are locked up (no opponent). |
| `Foosball-vs-v0` | Play against a frozen PPO model loaded from a checkpoint. Requires `--opponent`. |
| `Foosball-vsghost-v0` | Train against a hardcoded ghost opponent with curriculum levels 0-6. |
| `Foosball-ghostdemo-v0` | Demo a specific ghost level (1 env, play-only). Requires `--ghost_level`. |
| `Foosball-1player-vision-v0` | Vision mode, single player. Actor sees overhead RGB; critic sees state. |
| `Foosball-vs-vision-v0` | Vision mode, vs frozen opponent. |
| `Foosball-vsghost-vision-v0` | Vision mode, vs ghost curriculum. |

All environments share the same core: white team (agent) controls 4 rods (Keeper, Defense, Mid, Offense) with 8 actions (4 prismatic slide + 4 revolute spin). Observation is 41-dim: joint positions (16) + joint velocities (16) + ball position (3) + ball velocity (6).

## Vision mode (train from video)

Vision-mode tasks (`*-vision-v0`) feed the policy a stack of three 128×128 overhead RGB frames instead of the 41-dim state vector. The critic still sees the privileged state (asymmetric actor-critic), so value learning stays sample-efficient. They are trained with **skrl PPO**, not SB3, and require Isaac Sim's RTX renderer:

```bash
./scripts/clean_isaac_state.sh && \
  python IsaacLab/scripts/reinforcement_learning/skrl/train.py \
    --task Foosball-1player-vision-v0 --num_envs 256 --headless --enable_cameras
```

- **Always prefix training/play commands with `./scripts/clean_isaac_state.sh &&`.** Kit/carb leaks orphan `/dev/shm/carb-*` shared-memory segments on exit (even on a clean `simulation_app.close()`). The next launch then segfaults or freezes the terminal. The cleanup script sweeps orphans whose owning PID is dead — safe to run any time.
- `--enable_cameras` is required (TiledCamera needs the RTX renderer).
- `--num_envs 256` is the recommended ceiling; 1024 will likely OOM on a single GPU at 128×128 with frame-stack 3.
- Architecture: Nature-style CNN encoder (32→64→64 conv) + 512-FC actor head; 256×256 MLP critic on the state.

## train.py

```bash
python IsaacLab/scripts/reinforcement_learning/sb3/train.py --task <TASK_ID> [OPTIONS]
```

| Flag | Default | Description                                                                                                                 |
|------|---------|-----------------------------------------------------------------------------------------------------------------------------|
| `--task` | required | Environment task ID (see table above)                                                                                       |
| `--num_envs` | 1024 | Number of parallel environments                                                                                             |
| `--max_iterations` | from config | Total RL training iterations                                                                                                |
| `--checkpoint` | — | Resume training from a saved `.zip` checkpoint                                                                              |
| `--save_freq` | 10000 | Save a checkpoint every N agent steps                                                                                       |
| `--seed` | from config | Random seed                                                                                                                 |
| `--log_interval` | 100000 | Log metrics every N timesteps                                                                                               |
| `--opponent` | — | Path to frozen opponent `.zip` (for `Foosball-vs-v0`)                                                                       |
| `--ghost_level_steps` | 0 | Ghost curriculum: increase level every N env steps (0 = stay at min level). Can also set to an array (ie 100000,200000,...) |
| `--ghost_min_level` | 0 | Ghost curriculum: starting level (0-6)                                                                                      |
| `--video` | false | Record training videos                                                                                                      |
| `--video_length` | 200 | Length of recorded video in steps                                                                                           |
| `--video_interval` | 2000 | Steps between video recordings                                                                                              |

**Examples:**

```bash
# Basic single-player training
python train.py --task Foosball-1player-v0 --num_envs 1024 --max_iterations 10000

# Train against frozen opponent
python train.py --task Foosball-vs-v0 --opponent logs/sb3/Foosball-1player-v0/.../model.zip

# Train with ghost curriculum, starting at level 2, advancing every 200k env steps
python train.py --task Foosball-vsghost-v0 --ghost_min_level 2 --ghost_level_steps 200000

# Comprehensive training task
conda activate env_isaaclab && cd Projects/foosball/teo/Foosball_CU/
python IsaacLab/scripts/reinforcement_learning/sb3/train.py --task=Foosball-vsghost-v0 --ghost_level_steps 100000,400000,1200000,2400000,3200000,4200000 --headless --video --video_interval 10000 --video_length 500 --max_iterations 35000

# Resume after task
python IsaacLab/scripts/reinforcement_learning/sb3/train.py --task=Foosball-vsghost-v0 --checkpoint logs/sb3/Foosball-vsghost-v0/2026-04-13_11-56-41/model_2621440000_steps.zip --ghost_min_level 5 --ghost_level_steps 2200000 --headless --video --video_interval 50000 --video_length 500 --max_iterations 20000

# Play ghost
python IsaacLab/scripts/reinforcement_learning/sb3/play.py --task Foosball-ghostdemo-v0 --ghost_level 5 --checkpoint "/home/yw3809/Projects/foosball/teo/Foosball_CU/logs/sb3/Foosball-vsghost-v0/2026-04-13_11-56-41/model_2621440000_steps.zip"

```

Logs and checkpoints are saved to `logs/sb3/<task_name>/`. Monitor with TensorBoard:

```bash
tensorboard --logdir logs/sb3/
```

Key metrics: `rollout/ep_rew_mean`, `goal_scored_pct`, `opponent_goal_scored_pct` (vs/ghost modes), `ghost_level` (ghost mode).

## play.py

```bash
python IsaacLab/scripts/reinforcement_learning/sb3/play.py --task <TASK_ID> [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | required | Environment task ID |
| `--checkpoint` | latest | Path to model `.zip`. If omitted, uses the latest checkpoint from the log directory. |
| `--num_envs` | from config | Number of parallel environments |
| `--opponent` | — | Path to frozen opponent `.zip` (for `Foosball-vs-v0`) |
| `--ghost_level` | 0 | Ghost difficulty level 0-6 (for `Foosball-ghostdemo-v0`) |
| `--real-time` | false | Run at real-time speed |
| `--seed` | from config | Random seed |
| `--video` | false | Record video |
| `--video_length` | 200 | Length of recorded video in steps |
| `--use_last_checkpoint` | false | Use the last saved model instead of the best |

**Examples:**

```bash
# Watch trained agent vs static opponent
python play.py --task Foosball-1player-v0 --checkpoint logs/sb3/.../model.zip --real-time

# Watch agent vs frozen opponent
python play.py --task Foosball-vs-v0 --checkpoint model.zip --opponent opponent_model.zip

# Demo ghost level 5
python play.py --task Foosball-ghostdemo-v0 --ghost_level 5 --checkpoint model.zip --real-time
```

## Ghost Opponent

The ghost opponent is a hardcoded, non-AI controller for the black team with 7 difficulty levels (0-6). It provides a curriculum of increasing challenge for training the RL agent without requiring a pre-trained model checkpoint.

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

### Tuning

Ghost behavior is controlled by constants in `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/foosball2/ghost_opponent.py`:

- `ROD_X` — X positions of black rods (may need empirical tuning from USD geometry)
- `ROD_FIGURE_OFFSETS` — Y offsets of player figures on each rod
- `PRISMATIC_KP/KD` — PD gains for lateral tracking
- `REVOLUTE_KP/KD` — PD gains for rod rotation
- `SLOW_TRACKING_EFFORT / FAST_TRACKING_EFFORT` — Effort clamps for levels 4-5 vs 6
- `KICK_WINDUP_STEPS / KICK_STRIKE_STEPS` — Timing for kick wind-up and strike phases
- `TIMER_MIN / TIMER_MAX` — Range of physics steps a rod holds its up/down state (levels 1-2)

## Project Structure

The IsaacLab framework is vendored in `IsaacLab/`. Only the following files are project-owned:

- `IsaacLab/scripts/reinforcement_learning/sb3/train.py` — Training script
- `IsaacLab/scripts/reinforcement_learning/sb3/play.py` — Inference/playback script
- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/foosball2/` — Environment definition
  - `foosball_env.py` — `FoosballEnv` class and all env configs
  - `ghost_opponent.py` — Ghost opponent logic
  - `__init__.py` — Gym registrations
  - `agents/sb3_ppo_cfg.yaml` — PPO hyperparameters
- `IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/foosball.py` — Articulation configs (`FOOSBALL_CFG`, `FOOSBALL_VS_CFG`)
