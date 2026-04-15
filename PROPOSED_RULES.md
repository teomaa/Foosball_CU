# Proposed Rule-Based Agent v2

## Core Problem with Current Rules

Every rod does the same thing: track the ball and kick it forward. This means the goalie chases the ball instead of guarding the goal, the defense and midfield compete for the same ball instead of coordinating, and shots go in random directions instead of aiming at gaps. A competitive agent needs **role differentiation**, **prediction**, and **intent**.

---

## Design Principles

1. **Rods have roles, not a shared behavior.** The goalie blocks. The defense clears. The midfield controls. The attack scores.
2. **React to where the ball will be, not where it is.** Use ball velocity to predict arrival at each rod's Y coordinate.
3. **Shoot at gaps, not at random.** Use opponent rod positions (available in obs) to find open lanes.
4. **Control before shooting.** Trap the ball, then aim, then shoot — don't just flail at everything that passes by.
5. **Know which phase you're in.** Offense and defense require fundamentally different behavior.

---

## 1. Game Phase Detection

Determine the current phase from ball position and velocity. This drives which rods are active and what they do.

### Phase: DEFENDING
- **Condition:** Ball is in our half AND moving toward our goal (ball_vy toward own goal)
- **Priority:** Goal and Def block. Mid tracks. Attack holds neutral.

### Phase: NEUTRAL / CONTESTED
- **Condition:** Ball is near midfield OR has low velocity (stalled, bouncing)
- **Priority:** Nearest rod to ball takes possession. Others prepare for the next phase.

### Phase: ATTACKING
- **Condition:** Ball is in opponent's half OR moving toward opponent's goal
- **Priority:** Attack and Mid are active. Def and Goal hold defensive positions (don't chase).

The key behavioral change: **rods far from the ball stop tracking it and instead hold strategic positions.** Currently every rod chases the ball, which means nobody is ever in position for what comes next.

---

## 2. Ball Trajectory Prediction

Replace pure positional tracking (`ball_x`) with a predicted arrival point at each rod's Y level.

### Linear prediction
For a rod at `rod_y`, if the ball is at `(ball_x, ball_y)` with velocity `(vx, vy)`:
```
time_to_rod = (rod_y - ball_y) / vy      (only valid if vy is moving toward the rod)
predicted_x = ball_x + vx * time_to_rod
```

### Wall-bounce prediction
The table has walls at `x = ±x_max`. If `predicted_x` exceeds a wall, reflect it:
```
while predicted_x outside [-x_max, x_max]:
    reflect off the nearest wall
```
This handles one or two bounces, which covers the vast majority of real trajectories.

### When to use prediction vs. tracking
- **Ball moving toward the rod:** Use predicted arrival X. Slide to intercept.
- **Ball moving away from the rod:** Use current ball X. Just shadow it loosely.
- **Ball very close to the rod (within kick range):** Use current ball X. Prediction is unnecessary at close range.
- **Ball nearly stationary (|vy| < threshold):** Fall back to current ball X.

This one change alone should dramatically improve both defense (intercepting shots) and offense (getting into position before the ball arrives).

---

## 3. Role-Specific Behavior

### 3a. Goalkeeper

The goalie's job is to block shots, not to score.

**Slide behavior:**
- Always use predicted arrival X when the ball is moving toward our goal
- Constrain slide range tighter than the actuator allows — the goalie should stay centered, covering the goal mouth, not sliding to the extreme edges
- When ball is far away (opponent's half), hold center position instead of tracking

**Rotation behavior:**
- **Default:** Hold a partially angled position (~0.3-0.5 rad) that covers the most vertical space in front of the goal. Fully upright foosmen have gaps between them; a slight angle closes these gaps.
- **Clearing kick:** When the ball is very close and slow (trapped or rolling in), kick it forward hard to clear. Aim toward the side walls to prevent easy rebounds back toward goal.
- **Never cock back when ball approaches fast.** An incoming fast shot will pass through during the wind-up. Instead, hold the blocking angle and let the ball deflect.

### 3b. Defense Rod

Primary job: intercept balls coming through midfield and pass them forward to midfield or attack.

**Slide behavior:**
- Use predicted arrival X when ball is moving toward us
- When ball is on our side and slow, track it actively for possession

**Rotation behavior:**
- **Blocking posture** when ball is approaching fast (same as goalie — hold angle, don't cock)
- **Controlled pass** when in possession: Instead of a full-power forward kick, use a shorter rotation arc aimed toward the midfield rod. The goal is to place the ball near midfield, not blast it at max power.
- **Clearing kick** when under pressure (ball very close, opponents nearby): full-power kick toward a side wall to relieve pressure

### 3c. Midfield Rod

The transition rod — receives passes from defense, sets up the attack.

**Slide behavior:**
- When ball is approaching from our defense: predict arrival and intercept
- When ball is approaching from opponent's attack: predict and block (defensive role)
- When in possession: slide to align with a gap in the opponent's defense rod before passing/shooting

**Rotation behavior:**
- **Trap on arrival:** When receiving a pass from defense, rotate to pin/trap the ball rather than immediately kicking. A slight downward rotation as the ball arrives, holding it momentarily.
- **Pass forward:** After trapping (or if ball is slow and controlled), pass to attack rod. Use a moderate kick aimed at where the attack rod's foosmen can receive.
- **Block:** When ball approaches from opponent's side at speed, hold blocking angle.

### 3d. Attack Rod

The scoring rod. This is where shot intelligence matters most.

**Slide behavior:**
- When ball is far away: hold a position offset from center (don't track). Stay ready.
- When ball is approaching from midfield: predict arrival and intercept
- **Pre-shot positioning:** When in possession, slide to set up a specific shot angle before kicking

**Rotation behavior:**
- **Trap on arrival:** Pin the ball, don't immediately shoot
- **Aimed shot:** After trapping, choose a shot direction based on opponent gap analysis (section 4), then execute a pull-shot or push-shot (slide + kick simultaneously)
- **Snap shot:** If ball is arriving fast and already aimed well, skip trapping and redirect it with a timed kick

---

## 4. Opponent-Aware Shot Placement

The observation space includes opponent rod slide positions. Use these to find gaps.

### Gap detection
For the opponent's goalie and defense rods, compute the X positions of their foosmen (from their slide position + known foosman spacing). Identify the largest gap between adjacent foosmen, or between the outermost foosman and the wall.

### Shot targeting
Instead of a random offset, bias the kick slide-offset toward the largest detected gap:
```
gap_center = center of the widest gap in opponent's next rod
slide_offset = gap_center - ball_x    (adjusted for the kick angle)
```

Add a small random perturbation (±0.5) for unpredictability, but the base direction is intentional.

### Fallback
If no clear gap is detected (opponent is well-positioned), use one of:
- **Wall bank:** Aim toward a side wall so the ball bounces past the defenders at an angle they can't track
- **Straight power shot:** Rely on speed beating reaction time

---

## 5. Ball Trapping / Possession Control

Currently the agent has no concept of "having possession." The ball just bounces around and rods swing at it. Adding a trapping mechanic creates time to aim and coordinate.

### Trap detection
A rod "has possession" when:
- Ball is within a small radius of one of its foosmen (dx < 1.0, dy < 1.5)
- Ball velocity is low (|v| < some threshold)
- OR: the rod successfully performed a downward pin rotation

### Trap action
- Rotate the rod slightly downward (opposite of kick direction) to pin the ball against the table surface
- Hold slide position steady (don't track — you already have the ball)
- Maintain for a brief window (a few timesteps) while shot placement is computed

### Release
After the trap window, execute the aimed kick. The sequence is:
```
Ball arrives → Trap (pin) → Slide to aim → Kick → Return to neutral
```

This is slower than the current "kick everything immediately" approach, but produces far more accurate and effective shots.

---

## 6. Deliberate Passing

Currently, a kicked ball that happens to reach the next rod forward is accidental. Passing should be intentional.

### Pass targeting
When a rod kicks, it should aim toward a specific foosman on the next friendly rod forward:
```
target_x = position of nearest foosman on the next-forward rod
slide_offset = target_x - ball_x    (so the kick sends the ball toward that foosman)
```

### Receiving
The receiving rod should:
1. Pre-position at the predicted arrival X (using ball trajectory prediction)
2. Prepare a trap rotation (slight downward angle) to catch the pass
3. After trapping, either pass forward again or shoot

### Pass chain
The ideal sequence: **Def → Mid → Attack → Shoot**. Each stage:
- Def clears/passes to mid
- Mid traps, repositions, passes to attack
- Attack traps, reads opponent gaps, executes aimed shot

Not every play needs all three stages — if the attack rod is already in a good position when the ball arrives, skip the intermediate passes.

---

## 7. Improved Lane Clearing

The current lane clearing is reactive — rods lift when a back rod cocks. This can be smarter.

### Pre-emptive clearing
When a rod is about to receive a pass (predicted ball trajectory will reach it), rods between the passer and receiver should proactively lift, not wait for the cock phase.

### Selective clearing
Only lift rods that are actually in the ball's predicted path. If the ball will pass to the left, a foosman on the right side of an intermediate rod doesn't need to lift.

### Drop timing
Instead of dropping as soon as the ball passes the rod's Y, drop when the ball is safely past by a margin. Early drops can clip the ball.

---

## 8. Wall-Bank Shots

Intentional use of wall bounces for both passing and shooting.

### When to bank
- Opponent rod is well-positioned centrally (no direct gaps)
- Ball is near a side wall already (short angle to the wall)
- Passing laterally across a rod that's blocking the direct path

### Bank angle calculation
```
wall_x = nearest wall (±x_max)
distance_to_wall = |ball_x - wall_x|
reflected_target_x = 2 * wall_x - target_x    (mirror target across wall)
aim toward reflected_target_x
```

The kick offset and power are set so the ball hits the wall and deflects toward the desired target.

---

## Summary of Changes from Current Rules

| Aspect | Current | Proposed |
|--------|---------|----------|
| Rod behavior | All rods identical | Role-specific (goalie blocks, attack scores, etc.) |
| Slide tracking | Current ball_x | Predicted arrival X (with wall bounces) |
| When to kick | Whenever ball is near | Trap first, aim, then kick |
| Shot direction | Random offset | Aimed at gaps in opponent's rods |
| Defensive play | None | Goalie holds blocking angle, defense intercepts |
| Passing | Accidental | Deliberate pass chains with receiving traps |
| Lane clearing | Reactive to cock state | Pre-emptive based on predicted ball path |
| Wall use | None | Intentional bank shots and bank passes |
| Game awareness | None | Phase detection (attacking/defending/neutral) |
| Opponent awareness | None | Gap detection in opponent rod positions |

---

## Suggested Implementation Priority

If building incrementally, the order of impact is roughly:

1. **Ball trajectory prediction** — single biggest improvement; makes everything downstream work better
2. **Game phase detection + role differentiation** — stops the goalie from chasing the ball upfield
3. **Defensive goalie/defense behavior** — blocking angle, predicted interception
4. **Opponent-aware shot placement** — shooting at gaps instead of randomly
5. **Ball trapping** — control before shooting
6. **Deliberate passing** — coordinated offense
7. **Wall-bank shots** — advanced technique, lower priority
8. **Pre-emptive lane clearing** — refinement of existing system
