# Migration: Recompute `cartesian_delta` for legacy Vega trajectories

## Background

`VegaRobot.create_action_dict` (in `src/dexcontrol/core/vega/robot.py`) used to
populate the logged `cartesian_delta` field for `action_space="cartesian_velocity"`
inputs using a simple `cart_action * dt` formula. The actual motor delta the
server executes is produced by `_cartesian_velocity_to_delta`, which applies
norm clipping plus per-step `max_lin_delta` / `max_rot_delta` scaling. The two
diverge whenever the velocity command norm exceeds 1.0 or whenever
`max_*_delta` is not equal to `dt`.

Concretely, for every trajectory step where `action_space == "cartesian_velocity"`:

| Column                 | Source                                              | Correctness |
| ---------------------- | --------------------------------------------------- | ----------- |
| `cartesian_velocity`   | Original request action                             | OK          |
| `cartesian_delta`      | `cart_action * dt` (legacy code)                    | **Wrong**   |
| `delta_action`         | Built from the wrong `cartesian_delta`              | **Wrong**   |
| `joint_velocity`       | IK solve on the wrong `cartesian_delta`             | **Wrong**   |
| `joint_position`       | `current_joint_pos + joint_delta` from the IK above | **Wrong**   |
| `target_cartesian_delta` | `cartesian_velocity / gain`                       | OK          |
| `robot_state`          | Captured before the step                            | OK          |

The fix in `VegaRobot.create_action_dict` (this PR) makes the `cartesian_velocity`
branch call `_velocity_to_motor_delta` whenever the caller passes
`max_lin_delta` / `max_rot_delta` (which the server always does). New trajectories
produced after this fix have the correct values in every column.

This document describes how to recompute the affected columns for trajectories
that were collected **before** the fix.

## Eligibility

A trajectory step is eligible for recomputation when **all** of the following
hold:

1. `action_space == "cartesian_velocity"` for that step (the `target_cartesian_delta`
   branch was already correct and does not need migration).
2. `cartesian_velocity` was logged and is intact.
3. `robot_state.joint_positions` was logged for the same step (needed to redo
   IK).
4. The Vega control configuration that produced the data (specifically the
   values used to derive `max_lin_delta` / `max_rot_delta`) is known. These are
   functions of `control_hz` and the rotational sensitivity baked into the
   robot config; see `RobotEnvVegaService.__init__` in
   `src/dexcontrol/core/robotenv_vega/server.py` for the exact derivation
   (`_compute_cartesian_delta_limits`, around line 248).

If any of the above is missing, the affected columns cannot be reconstructed
exactly. Drop the row or mark it as unmigratable.

## Recomputation procedure

For each eligible step:

1. **Recover `max_lin_delta` and `max_rot_delta`** for the control config that
   was active when the data was collected. The values follow
   `_compute_cartesian_delta_limits(control_hz, rot_sensitivity)`:
   ```python
   baseline_hz = 20.0
   scale = float(np.clip(1.0 - (control_hz - baseline_hz) / 80.0, 0.1, 1.5))
   max_lin_delta = 0.075 * scale
   max_rot_delta = 0.3 * scale * rot_sensitivity
   ```
   Use the values that match the data source. When `rot_sensitivity` was not
   recorded, default it to `1.0` (the standard config).

2. **Recompute `cartesian_delta`** via the same transform the server uses:
   ```python
   def velocity_to_motor_delta(cv, max_lin_delta, max_rot_delta):
       cv = np.asarray(cv, dtype=np.float64).copy()
       lin = cv[:3]
       rot = cv[3:6]
       lin_norm = float(np.linalg.norm(lin))
       rot_norm = float(np.linalg.norm(rot))
       if lin_norm > 1.0:
           lin = lin / lin_norm
       if rot_norm > 1.0:
           rot = rot / rot_norm
       out = np.empty(6, dtype=np.float64)
       out[:3] = lin * max_lin_delta
       out[3:6] = rot * max_rot_delta
       return out
   ```
   This mirrors `_cartesian_velocity_to_delta` at the time of writing; if that
   function evolves, the migration script should be updated to track it.

3. **Rebuild `delta_action`** as `np.concatenate([cartesian_delta, [gripper_action]])`,
   using the same `gripper_action` already stored in the trajectory (or extracted
   from the original action vector).

4. **Recompute `cartesian_position`** by adding the new `cartesian_delta` to
   `robot_state.cartesian_position` (the same pose-addition the original code
   does).

5. **Re-solve IK** to recover `joint_position` and `joint_velocity`. Use the
   same IK solver Vega ships with, seeded by `robot_state.joint_positions` for
   the step. On IK failure, keep `joint_position = robot_state.joint_positions`
   and `joint_velocity = [0.0] * 7`, mirroring the original code's fallback.

6. **Leave `cartesian_velocity` and `target_cartesian_delta` untouched** —
   they were already correct.

## Where to add the migration script

Add the script under
`utils/s3-data-migration/configint-raw/recorder/fix-vega-cartesian-delta/`
(beside `fill-missing-action-columns/`). The directory layout used by the other
fix-up jobs there gives you a template:

```
fix-vega-cartesian-delta/
  src/
    transform.py       # the velocity_to_motor_delta + IK loop
    run.py             # iterate over parquet files
  README.md
```

Reuse the existing parquet loader / writer helpers. The IK solver dependency
must come along — either by adding `custom_dexcontrol` as a dependency of the
migration job, or by re-implementing the small slice of IK that
`_solve_cartesian_delta` uses.

## Validation

For a small sample of pre-fix trajectories that were replayed on the same
robot configuration:

1. Run the migration to produce a corrected `cartesian_delta` column.
2. Replay the corrected `cartesian_delta` through the server in `cartesian_delta`
   action_space and confirm the resulting end-effector trace matches the
   trajectory's original `robot_state.cartesian_position` to within IK
   tolerance.

If the validation diverges by more than a few millimetres / a few milliradians,
the assumed `max_*_delta` values were wrong for that data source — recheck the
source's `control_hz` / `rot_sensitivity`.

## Scope of impact

- Trajectories collected with `action_space="cartesian_velocity"` on Vega
  before the `VegaRobot.create_action_dict` fix.
- New trajectories produced after the fix are already correct and do not need
  migration.
- Franka data is unaffected — Franka's `create_action_dict_with_joint_feedback`
  does its own IK-based `cartesian_velocity_to_delta`, which has no analogous
  `max_*_delta` clamping step.
