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

## Two lanes

- **obs lane** — an `robot-obs` poll thread reads the current observation on a
  clock and publishes it. Vega computes obs only inside `_create_observation`, so
  the bridge must drive it on a clock: otherwise teleop (which needs obs to
  compute a delta) and obs (driven by the resulting action's `Step`) deadlock at
  startup. This lane breaks that cycle, so obs streams independently of actions.
- **action lane** — a thread subscribes `robot-action` and replays each frame
  through the service's own `Step` (reusing its teleop-gain / frame-transform /
  interpolation logic), retrying until the source opens.

## Layout

| Module | Role | Depends on |
|---|---|---|
| `robot_obs.py` | `robot-obs` channel layout + observation→step-dict projection | loop-sdk only |
| `robot_action.py` | `robot-action` decode (`<arm>.action.<space>[i]` → action vector) | loop-sdk only |
| `obs_publisher.py` | `RobotObsPublisher` — wraps a loop-sdk `RobotStepSender` | loop-sdk only |
| `action_consumer.py` | `ArmActionBackend` (per-arm decode→Step) + `RobotActionConsumer` (one source → N backends) | loop-sdk only |
| `lanes.py` | `run_obs_poll` + `run_action_lane` (the two thread bodies) | loop-sdk only |
| `source_server.py` | `_LockedStepService` + `LoopBridge` (N arm services → one robot-obs) + `serve_with_loop` (single) / `serve_dual_arm` (bimanual) | dexcontrol + loop-sdk |
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
