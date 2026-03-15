# Foosball RL Project

Reinforcement learning environment for training a foosball-playing AI agent, built on NVIDIA IsaacLab (Isaac Sim).

## Scope - What to look at

The vast majority of this repo is the IsaacLab framework (vendored in `IsaacLab/`). **Do not read or modify IsaacLab files** unless they are listed below as project-owned.

### Project-owned files

**Top-level entry points:**
- `sac_agent_entry.py` - SAC training entry (v1, protagonist-antagonist, image-based)
- `sac_agent_entry_v2.py` - SAC training entry (v2, single-player, full-information)

**IsaacLab foosball task (the main RL environment):**
- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/foosball2/` - Task definition
  - `foosball_env.py` - Main environment (`FoosballEnv`, `FoosballEnvCfg`). This is the core file.
  - `__init__.py` - Gym registration (env id: `Foosball-1player-v0`)
  - `agents/` - RL algorithm config files (PPO configs for rl_games, skrl, sb3, rsl_rl)

**IsaacLab foosball asset:**
- `IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/foosball.py` - `FOOSBALL_CFG` articulation config (joint definitions, actuators)
- `IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/__init__.py` - Exports foosball config

**Standalone copies / setup scripts (STALE — do not edit):**
- `Isaac_Lab_Files/` - Old standalone copies of foosball files + USD meshes + setup scripts. These are **out of sync** with the IsaacLab tree copies and should be ignored. The IsaacLab tree copies (above) are the ones used by training and are the only ones to edit.
  - `setup.sh` - Was used to copy files into the IsaacLab tree; no longer needed since we edit the tree directly
  - `Foosball_Meshes.usd`, `foosball_no_ball.usd` - USD mesh assets (still referenced by the tree)
  - `rule_based_foosball_agent.py`, `rule_based_main.py` - Rule-based baseline agent

### Files to ignore

- `IsaacLab/` (everything not listed above) - Framework code, not ours
- `logs/`, `outputs/` - Training artifacts (gitignored)

## Architecture

- **Environment:** IsaacLab `DirectRLEnv` subclass. Single-player (white team) controls 4 rods (Keeper, Defense, Mid, Offense) each with prismatic (slide) + revolute (spin) joints = 8 action dims. Black team rods are locked in place. Ball is a rigid sphere.
- **Observation space (41):** Joint positions (16) + joint velocities (16) + ball position (3) + ball velocity (6)
- **Action space (8):** Effort targets for 4 prismatic + 4 revolute white joints, clamped to [-1, 1] then scaled
- **Reward:** Sparse goal reward (+500 for scoring) + continuous distance-to-goal shaping penalty
- **Termination:** Ball off table, ball too high, goal scored, or episode timeout (10s)
- **Simulation:** 120 Hz physics, decimation=2, 1024 parallel envs

## Key conventions

- Uses `stable_baselines3` for SAC agents, IsaacLab RL wrappers (rl_games, skrl, sb3, rsl_rl) for PPO
- Joint naming: `{Row}_{Color}_{JointType}` e.g. `Keeper_W_PrismaticJoint`, `Defense_B_RevoluteJoint`
- USD asset path in `foosball.py` is hardcoded and updated by `setup.sh`

## Editable install: editing IsaacLab source files

The `isaaclab_tasks`, `isaaclab_assets`, and `isaaclab_rl` packages are pip-installed in **editable mode** from `/home/yw3809/Projects/foosball/IsaacLab/`, NOT from this repo's `IsaacLab/` copy. When training scripts run, Python imports from the editable install location.

**When editing any project-owned file under `IsaacLab/source/`**, you must apply the change to **both** locations:

1. This repo's copy: `./IsaacLab/source/isaaclab_tasks/...` (for version control)
2. The editable install: `/home/yw3809/Projects/foosball/IsaacLab/source/isaaclab_tasks/...` (what actually runs)

If you only edit the repo copy, the change will have no effect at runtime.
