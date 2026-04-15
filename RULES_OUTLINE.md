# Rule-Based Foosball Agent: Current Playing Rules

## Overview

Each team controls 4 rods (goal, def, mid, attack), each with 2 degrees of freedom: **slide** (lateral movement) and **rotation** (kick). The rule-based agent produces an 8-dimensional action vector `[slide_goal, rot_goal, slide_def, rot_def, slide_mid, rot_mid, slide_attack, rot_attack]` every timestep.

The logic runs identically for both teams (protagonist/antagonist), with Y-axis directions flipped. Yellow attacks toward +Y, black attacks toward -Y.

---

## 1. Slide Control (Lateral Tracking)

**Goal:** Move each rod so that its nearest foosman aligns with the ball's X position.

**How it works:**
- For each rod, find the foosman closest to the ball (Euclidean distance in XY)
- Compute the X-error between that foosman and the ball: `ball_x - guy_x`
- Convert to slide-joint space using a cached `dx/ds` ratio (computed once at init via finite difference)
- Apply an aggressive gain: `delta_s = slide_gain * (ball_x - guy_x) / dx_ds`
- Clip the per-step delta to `[-slide_step_clip, +slide_step_clip]`
- Clip the final target to the rod's actuator range

**Parameters:**
| Param | Value | Effect |
|---|---|---|
| `slide_gain` | 5.0 | Multiplier on ideal slide delta — higher = more aggressive tracking |
| `slide_step_clip` | 7.0 | Max slide change per timestep (prevents overshoot) |

---

## 2. Cock-and-Kick (Rotation)

**Goal:** Wind up the rod when the ball is nearby, then strike it forward.

**Two-phase approach:**

### Phase 1: Cocking (wind-up)
- **Trigger:** Ball is in a "ready region" relative to the nearest foosman:
  - X distance: `|ball_x - guy_x| <= kick_x_close` (2.0 units)
  - Y distance (ball ahead of foosman): `kick_y_front_min <= dy <= kick_y_front_max` (-1.2 to 9.0 units)
- **Action:** Rotate the rod backward by `kick_back_angle` (1.95 rad)
- **Persistent state:** Once cocked, the rod _stays_ cocked across timesteps until the ball passes the rod's Y coordinate (i.e., the shot was taken or the ball moved on)

### Phase 2: Forward strike
- **Trigger:** Ball is in a tighter "contact region":
  - X distance: `|ball_x - guy_x| <= kick_x_contact` (0.3 units)
  - Y distance: `0 <= dy <= kick_y_contact_max` (0 to 1.25 units)
- **Action:** Rotate forward by `kick_forward_angle` (3.90 rad, ~2x the cock-back angle)
- **Angled shot:** On strike, the slide target gets a random offset in `[-kick_offset_max, +kick_offset_max]` (5.0 units) to vary shot angle

### Fast-approach shortcut
- If the ball is approaching from the front at high speed (`ball_vy > front_fast_vy_thresh`, 9.0 units/s), **skip cocking entirely** and go straight to forward strike on contact
- This handles situations where there's no time to wind up

**Parameters:**
| Param | Value | Effect |
|---|---|---|
| `kick_x_close` | 2.0 | X-distance threshold to begin cocking |
| `kick_x_contact` | 0.3 | X-distance threshold for forward strike |
| `kick_y_front_min` | -1.2 | Min Y-distance for cocking window (allows slightly behind) |
| `kick_y_front_max` | 9.0 | Max Y-distance for cocking window |
| `kick_y_contact_max` | 1.25 | Max Y-distance for contact strike |
| `kick_back_angle` | 1.95 rad | Cock-back rotation magnitude |
| `kick_forward_angle` | 3.90 rad | Forward strike rotation magnitude |
| `kick_sign` | 1.0 | Flips rotation direction (set to -1.0 if mesh is reversed) |
| `kick_offset_max` | 5.0 | Random slide offset on strike for angle variation |
| `front_fast_vy_thresh` | 9.0 | Ball speed threshold for skipping the cock phase |

**State diagram:**
```
Idle (rot=0) ──[ball in ready region]──> Cocked (rot=back_angle)
                                              │
                              [ball in contact region]
                                              │
                                              v
                                     Strike (rot=forward_angle)
                                              │
                                   [ball passes rod Y]
                                              │
                                              v
                                         Idle (rot=0)
```

---

## 3. Lane Clearing

**Goal:** When a rear rod is kicking, lift the rods in front of it out of the ball's path so it can pass through.

**How it works:**
- Rods are ordered back-to-front: goal → def → mid → attack
- When **any rod behind** the current rod is in a cocked/kicking state, the current rod lifts by rotating to `clear_lane_angle` (-2.0 rad)
- The rod stays lifted until the ball passes its Y coordinate (ball has moved through)
- A lifted rod will **not** re-lift once the ball has already passed it

### Opportunistic kick from lane-clear position
- If a rod is currently lifted for lane clearing and the ball approaches it from behind (coming from the kicking rod):
  - Contact condition: `|ball_x - guy_x| <= kick_x_contact` and `0 <= dy_back <= kick_y_contact_max`
  - The rod drops from lane-clear and executes a forward strike instead
  - After this opportunistic kick, the rod returns to normal (lane-clear state resets)

**Parameters:**
| Param | Value | Effect |
|---|---|---|
| `clear_lane_angle` | -2.0 rad | Rotation angle for lifted rods |

---

## 4. Default Behavior

When none of the above conditions apply:
- **Slide:** Track ball X (rule 1 always runs)
- **Rotation:** Return to 0 (neutral/upright position)

---

## Execution Order (per timestep)

1. **First pass** — for each rod (goal → def → mid → attack):
   - Compute nearest foosman position
   - Compute slide target (always tracks ball X)
   - Evaluate cock/kick state machine (check ready region, contact region, fast-approach)
   - Record whether this rod is cocked

2. **Second pass** — for each rod (goal → def → mid → attack):
   - Check if any rod behind it is cocked → lift for lane clearing
   - Check if a lifted rod should drop (ball passed) or opportunistically kick
   - Write final slide and rotation targets to the action vector

3. **Output:** 8-dimensional action vector, all values clipped to actuator ranges

---

## What the Rules Do NOT Handle

- No explicit defensive positioning (goalie doesn't prioritize blocking shots)
- No ball-velocity-based prediction (tracking is purely positional, not predictive)
- No passing strategy (rods don't coordinate to advance the ball deliberately)
- No opponent awareness (doesn't track or react to opponent rod positions)
- Contact jitter: small random perturbation (±0.5) is added to slide on contact, but there's no structured dribbling or ball-control logic
