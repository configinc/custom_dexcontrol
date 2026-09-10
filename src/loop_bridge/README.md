# Vega Robot Node

One external `loop.robot@1` Node controls both arms of a Vega through the existing
DexControl IK, interpolation, filtering, and gripper implementation.

## Install and run

From the repository root, with Git, uv, a C++ compiler, CMake, and GitHub access:

```bash
./install.sh --extra robotiq
export LOOP_NODE_GRAPH_NODE_ENDPOINT=tcp/loop-host:7448
./run.sh --node-id robot --gripper-type robotiq \
  --robotiq-comport-left /dev/ttyUSB0 \
  --robotiq-comport-right /dev/ttyUSB1
```

The installer creates a Python 3.12 `.venv`, downloads pinned SDK/Node sources to
`third_party`, and installs the selected gripper. Use `--extra sr-gripper` for SR
EtherCAT grippers, or omit the gripper extra for built-in hands. With SR, the
per-arm port arguments are network interface names. Ansible configures the
required EtherCAT permissions.

`--loop-endpoint` overrides `LOOP_NODE_GRAPH_NODE_ENDPOINT`; the fallback is
`tcp/127.0.0.1:7448`. This is the Loop connection. DexComm's `ROBOT_NAME` and
`ZENOH_CONFIG` still configure communication with the physical Vega.

`./run.sh --help` lists the hardware and controller options. Match `--node-id` to
the Robot Node ID in the Loop Cell Config. Both Robotiq and SR require distinct
ports for the left and right grippers.

## Lifecycle

| Event | Behavior |
| --- | --- |
| Process start | Register with Loop in IDLE; no arm service is opened yet. |
| Start | Open hardware on the first Start, resume control, and publish an initial observation. |
| Stop | Stop observation/control workers, discard pending arm/gripper commands, and hold the current arm position. Keep the hardware connection. |
| Fault | Pause both arms and report the failure to Loop. Reset Fault returns to IDLE; Start resumes operation. |
| Shutdown / process exit | Stop workers and close grippers and the shared Vega connection. |

The existing hardware initialization, including control-mode/head initialization,
happens on the first Start. Later Starts reuse the connection.

## Data and commands

| Port | Data |
| --- | --- |
| `robot_observation` output | `left/right.cartesian_position`, gripper position, joints, velocities, torques, and wrench |
| `action_command` input | `left/right.target_cartesian_delta`: float64 `[6]`; `left/right.gripper_position`: scalar |
| `robot_command` request/reply | Home both arms through the existing reset paths; return failure if either arm fails. |

For an action, the Node reads both arm observations once and passes those same
pre-action states to each arm's Step. After dispatch it publishes that observation
with the received action and per-arm action details. A failed Step faults the Node.
Loop's Main Controller builds the final ControlStep for recording.

When actions are idle, state-only observations resume after two heartbeat periods
(default heartbeat: 20 Hz). The input keeps the latest pending action rather than
a backlog of old targets.

## Development

```bash
./install.sh --extra robotiq --extra dev
.venv/bin/python -m pytest -q tests/loop_bridge
```

See [deployment](../../ansible/README.md). Legacy per-arm gRPC entrypoints remain
available for existing clients; the Loop deployment runs the single Robot Node.
