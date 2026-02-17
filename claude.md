# Foosball CU

MuJoCo-simulated foosball table with AI agents. The simulation uses a physically realistic table with 4 rods per player (goal, def, mid, attack), each with 2 DOF (linear slide + rotation), and a free-body ball.

## Current Focus: Rule-Based Baseline

The active work is on a **geometry-based rule agent** — no ML training. The goal is a competent foosball player using simple heuristics (track ball, cock-and-kick, clear lanes). Entry point: `sac_agent_entry_rule_based.py`.

Despite the SAC infrastructure in the entry point, the actual rule-based logic lives entirely in the environment's hardcoded action methods. The SAC agent/training engine are scaffolding that may be replaced or simplified.

## Architecture

```
sac_agent_entry_rule_based.py          # Entry point (CLI flags: --protagonist-strategy, --antagonist-strategy)
├── ai_agents/v2/gym/
│   ├── full_information_protagonist_antagonist_gym_rule_based.py  # FoosballEnv (delegates to strategies)
│   └── strategies/
│       ├── __init__.py               # make_strategy() factory, registry
│       ├── base_strategy.py          # FoosballStrategy ABC
│       ├── basic_strategy.py         # Original cock-and-kick logic (extracted from env)
│       └── advanced_strategy.py      # Smarter rules: phases, roles, prediction, trapping, gap aiming
├── ai_agents/common/train/
│   ├── impl/
│   │   ├── single_player_training_engine_rule_based.py    # HardcodedAntagonist wrapper
│   │   ├── sac_agent.py                                   # SB3 SAC wrapper (7-layer MLP, 3000 neurons)
│   │   └── generic_agent_manager.py                       # Agent lifecycle
│   └── interface/                                         # ABCs: TrainingEngine, FoosballAgent, AgentManager
└── foosball_sim/v2/foosball_sim.xml                       # MuJoCo model (SDF collisions, meshes)
```

## Environment Details

- **Observation space**: 38 floats — ball pos (3), ball vel (3), rod slides (8), rod slide vels (8), rod rotations (8), rod rotation vels (8)
- **Action space**: 8 floats per player — 4 rods × (slide, rotation)
- **Slide ranges**: goal [-10,10], def [-20,20], mid [-7,7], attack [-12,12]
- **Rotation range**: [-2.5, 2.5] rad for all rods
- **Table Y dimension**: 65 units per side (130 total). Protagonist defends -Y, attacks +Y.
- **Termination**: ball out of Z bounds, no progress (>15 steps), ball stagnant (vel < 0.5 for >10 steps), max 40 timesteps, or goal scored.

## Reward Function

```
reward = victory(1000) + loss(-1000) + inverse_distance_to_opponent_goal(300 - dist) + control_cost(-0.005 * |action|)
```

## Pluggable Strategy System

Playing logic is encapsulated in **strategy objects** (see `ai_agents/v2/gym/strategies/`). Each player selects a strategy independently via CLI flags. The env delegates `protagonist_all_rods_action_toward_ball` and `antagonist_all_rods_action_toward_ball` to the respective strategy's `compute_action(obs)`.

### BasicStrategy (`basic_strategy.py`)
The original heuristics, extracted from the env:
1. **Slide control**: Move each rod so its nearest foosman aligns with ball_x (gain=5.0, clipped to ±7.0)
2. **Cock-and-kick**: When ball is in ready region, cock back (1.95 rad). On contact, kick forward (3.90 rad). Fast-approaching balls skip cocking.
3. **Lane clearing**: Rods in front of a kicking rod rotate out of the way (-2.0 rad). Lanes drop once ball passes.
4. **Persistent state**: `rod_cocked_state`, `rod_cocked_y`, `lane_clear_state` carry across timesteps.

Tunable params are instance attributes on BasicStrategy (kick_x_close, kick_back_angle, slide_gain, etc.).

### AdvancedStrategy (`advanced_strategy.py`)
Smarter agent implementing PROPOSED_RULES.md:
- **Game phase detection**: DEFENDING / NEUTRAL / ATTACKING based on ball position + velocity
- **Role-specific behavior**: goalkeeper (block+center bias), defense (intercept+pass), midfield (trap+pass), attack (trap+aimed shot)
- **Ball trajectory prediction**: linear extrapolation with wall-bounce reflection
- **Opponent gap analysis**: reads opponent foosman X positions, finds widest gap for aimed shots
- **Trap-then-act**: pin ball for N steps, then kick/pass (controlled possession)
- Falls back to standard cock-and-kick when conditions don't warrant advanced behavior

## Running

```bash
# Test basic vs basic (default, matches original behavior)
python sac_agent_entry_rule_based.py -t

# Advanced protagonist vs basic antagonist
python sac_agent_entry_rule_based.py -t --protagonist-strategy advanced --antagonist-strategy basic

# Basic vs advanced
python sac_agent_entry_rule_based.py -t --protagonist-strategy basic --antagonist-strategy advanced

# Train SAC against rule-based antagonist (currently set to 1 epoch, 100 timesteps for debugging)
python sac_agent_entry_rule_based.py
```

## Dependencies

- MuJoCo (with SDF collision plugin)
- gymnasium
- stable-baselines3 (SAC)
- numpy

## Key Conventions

- Yellow team (`y_`) = protagonist, attacks toward +Y
- Black team (`b_`) = antagonist, attacks toward -Y
- Rod order: goal (0), def (1), mid (2), attack (3)
- Actuator naming: `{y|b}_{goal|def|mid|attack}_{slide|rot}`
- The env caches MuJoCo body/joint/actuator IDs at init for performance
