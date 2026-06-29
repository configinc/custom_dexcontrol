# loop_bridge — Vega ↔ Loop Source Bus

Bridges Vega's RobotEnv to Loop's Source Bus **in-process** with the RobotEnv
server: it publishes observations as a `robot-obs` source and executes a
`robot-action` source it consumes from the bus by replaying each action through
the RobotEnv `Step` path. The plain RobotEnv gRPC contract (Step/Reset/...) is
unchanged — this only adds the bus I/O edge.

This package lives at the top level (not under `dexcontrol/`) on purpose: it is
our addition on top of the upstream-synced `dexcontrol` tree, and keeping it
separate (a) minimizes upstream-sync merge conflicts and (b) lets the bus logic
be unit-tested without the heavy `dexcontrol` import.

## The loop

The control loop lives in loop-sdk's `LoopRobotClient.run` (timing, connect/disconnect,
the poll/drain/publish wiring). `LoopBridge` just supplies its device callbacks and,
because the bridge runs **in-process** with the RobotEnv gRPC server, starts `run()` on
a daemon thread while the main thread serves `Step`:

- `_read_obs` (`publish_obs_callback`) reads the current observation on the clock and
  returns it to publish. Vega computes obs only inside `_create_observation`, so the
  bridge drives it on a clock — that's what bootstraps obs from tick 0 with no input, so a
  relative-delta teleop can start (obs gated on an action's `Step` would never produce the
  first sample).
- `_apply_action` (`poll_action_callback`) decodes each arm's vector from the raw
  `robot-action` payload and replays it through the service's own `Step` (reusing its
  teleop-gain / frame-transform / interpolation logic).
- `_apply_command` (`drain_commands_callback`) homes each arm on a `HOME` command.

## Layout

| Module | Role | Depends on |
|---|---|---|
| `robot_obs.py` | `robot-obs` channel layout + observation→step-dict projection | loop-sdk only |
| `robot_action.py` | `robot-action` decode (`<arm>.action.<space>` → action vector) + `HOME` | loop-sdk only |
| `obs_publisher.py` | `merge_observations` — merge each arm's obs into one `robot-obs` dict | loop-sdk only |
| `source_server.py` | `_LockedStepService` + `LoopBridge` (N arm services → callbacks for `LoopRobotClient.run`, on a thread) + `serve_with_loop` (single) / `serve_dual_arm` (bimanual) | dexcontrol + loop-sdk |
| `__main__.py` | CLI launcher | the above |

Every module except `source_server` has no `dexcontrol` import, so the tests in
`tests/loop_bridge/` run with only loop-sdk installed.

## Install & run

loop-sdk is an internal package (not on PyPI); install via the `loop` extra:

```bash
pip install -e '.[loop]'

# single-arm:
python -m loop_bridge --loop-addr loop-host:50051 \
    --arm-side left --gripper-type robotiq --robotiq-comport /dev/ttyUSB0

# bimanual (ONE Vega, both arms → one robot-obs robot0+robot1):
python -m loop_bridge --loop-addr loop-host:50051 --dual-arm

# obs-only (no motion) — useful for first bring-up:
python -m loop_bridge --loop-addr loop-host:50051 --no-action ...
```

Vega supports Python 3.10–3.13, so loop-sdk runs directly in the RobotEnv server
process (in-process pattern). Per the robot source contract, a bimanual Vega is
ONE robot: `serve_dual_arm` builds one `Robot` and drives both arms over it (two
per-arm services sharing the unit), presenting one `robot-obs` (`robot0`+`robot1`)
and one `robot-action`. Bimanual needs the built-in per-arm grippers (a single
serial gripper can't be shared across both arms).

## Wire format

Channels follow the RCI convention (named, namespaced by arm prefix, default
`robot0`):

- **obs** `robot-obs` — `<arm>.observation.state.<field>[i]`:
  - **CORE** `joint_positions[0..6]` (rad), `gripper_position` (normalized 0–1),
    `cartesian_position[0..5]` (xyz m, rpy rad)
  - **AUX** `joint_velocities[0..6]` (rad/s), `joint_torques_computed[0..6]` (N·m),
    `wrench_state[0..5]` (force N, torque N·m)
- **action** `robot-action` — `<arm>.action.<space>[i]`, e.g.
  `target_cartesian_delta` (6 cartesian terms + 1 gripper).

## Tests

```bash
PYTHONPATH=src pytest tests/loop_bridge   # loop-sdk in env; dexcontrol not required
```
