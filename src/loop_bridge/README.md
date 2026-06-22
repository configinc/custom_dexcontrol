# loop_bridge — Vega → Loop Source Bus

Publishes Vega's RobotEnv observations onto Loop's Source Bus as a `robot-obs`
source, **in-process** with the RobotEnv server. Same values, same timing as the
RobotEnv `Step`/`Reset` RPC already returns — only an extra output edge.

This package lives at the top level (not under `dexcontrol/`) on purpose: it is
our addition on top of the upstream-synced `dexcontrol` tree, and keeping it
separate (a) minimizes upstream-sync merge conflicts and (b) lets the bus logic
be unit-tested without the heavy `dexcontrol` import.

## Layout

| Module | Role | Depends on |
|---|---|---|
| `robot_obs.py` | `robot-obs` channel layout + observation→vector flatten | loop-sdk only |
| `obs_publisher.py` | `RobotObsPublisher` — wraps a loop-sdk `SourceProducer` | loop-sdk only |
| `source_server.py` | `LoopVegaRobotEnvService` (subclass) + `serve_with_loop()` | dexcontrol + loop-sdk |
| `__main__.py` | CLI launcher | the above |

`robot_obs` and `obs_publisher` are pure and have no `dexcontrol` import, so the
tests in `tests/loop_bridge/` run with only loop-sdk installed.

## Install & run

loop-sdk is an internal package (not on PyPI); install via the `loop` extra:

```bash
pip install -e '.[loop]'
python -m loop_bridge \
    --loop-addr loop-host:50051 \
    --arm-side left --gripper-type robotiq --robotiq-comport /dev/ttyUSB0
```

Vega supports Python 3.10–3.13, so loop-sdk runs directly in the RobotEnv server
process (in-process pattern). The plain `dexcontrol...server` RobotEnv contract
(Step/Reset/...) is unchanged; this only adds `robot-obs` publishing alongside it.

## Channel layout (`robot-obs`)

`values[i]` aligns to `build_obs_channels()[i]`:

- **CORE** — `joint_positions[0..6]` (rad), `gripper_position` (normalized 0–1),
  `cartesian_position[0..5]` (xyz m, rpy rad)
- **AUX** — `joint_velocities[0..6]` (rad/s), `joint_torques_computed[0..6]` (N·m),
  `wrench_state[0..5]` (force N, torque N·m)

## Tests

```bash
PYTHONPATH=src pytest tests/loop_bridge   # loop-sdk in env; dexcontrol not required
```
