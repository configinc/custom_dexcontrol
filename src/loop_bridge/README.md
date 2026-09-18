# Vega Robot Node

One external `loop.robot@1` Node controls both arms of a Vega through the existing
DexControl IK, interpolation, filtering, and gripper implementation.

## Install and run

From the repository root, with Git, uv, a C++ compiler, CMake, and GitHub access:

```bash
./install.sh --extra robotiq
./run.sh --node-id robot
```

The installer creates a Python 3.12 `.venv`, installs the published `loop-sdk`
0.6.x and `loop-node` 0.3.x wheels from PyPI using `uv.lock`, and installs the
selected gripper. Loop must use `loop-node` 0.3.x (control protocol version 2);
update Loop and external Nodes together. Use `--extra sr-gripper` for SR
EtherCAT grippers, or omit the gripper extra for built-in hands. With SR, the
per-arm Node Config devices are network interface names. Ansible configures the
required EtherCAT permissions.

The deployment constraints in `pyproject.toml` and `uv.lock` keep DexMotion at
0.4.1, `dexmate-urdf` at 0.8.3, and NumPy at 2.4.6 to match the working RCI
runtime's IK library and robot model. The NumPy pin applies to Python 3.11 and
later; Python 3.10 retains a compatible NumPy release. Use the installer to
apply these constraints; a plain `pip install` does not apply uv constraints.

`--loop-endpoint` overrides `LOOP_NODE_GRAPH_NODE_ENDPOINT`; the fallback is
`tcp/192.168.5.17:7448`, Teleop's fixed wired address on the Vega network.
Teleop's Loop listener uses `tcp/0.0.0.0:7448`; no manual endpoint setting is
needed for this layout. This is the Loop connection. DexComm's `ROBOT_NAME` and
`ZENOH_CONFIG` still configure communication with the physical Vega.

Match `--node-id` to the Robot Node ID in the Loop Cell Config.

## Node Config

```json
{
  "frame_type": "vega-1-pro_torso_frame_v1",
  "action_frequency_hz": 20,
  "control_frequency_hz": 200,
  "gripper_type": "robotiq",
  "left_gripper_device": "/dev/ttyUSB0",
  "right_gripper_device": "/dev/ttyUSB1"
}
```

| Field | Meaning |
| --- | --- |
| `frame_type` | Arm startup and Home preset: `vega-1-pro_torso_frame_v1` (default) or `vega-1-pro_torso_frame_v2` |
| `action_frequency_hz` | Main Controller’s action frequency, used to interpret action deltas and velocities; default 20 Hz |
| `control_frequency_hz` | Common tick for both arms’ interpolation commands and observation sampling; default 200 Hz |
| `gripper_type` | `default` (built-in), `robotiq`, or `sr_gripper` |
| `left_gripper_device`, `right_gripper_device` | Distinct serial paths for Robotiq or network interfaces for SR; ignored for built-in grippers |

For SR, set both devices to EtherCAT interfaces such as `enp1s0` and `enp2s0`.
The selected driver must be installed; Ansible installs both optional drivers.
The existing deployed IK and interpolation tuning remain internal defaults.
Actions use `target_cartesian_delta` and
normalized gripper `position`. No DS Layout Unit Config lookup is required.

In existing Cell Configs, rename `control_hz` to `action_frequency_hz` and replace
`observation_frequency_hz` with `control_frequency_hz` (normally 200). The latter
now controls motor command timing as well as observation sampling.

The two frame presets use the joint targets from the existing Interface's
`frame.yaml`. On hardware initialization and Home, each arm opens its gripper,
moves through the existing intermediate pose, then moves to the selected target.
`frame_type` selects arm poses only; it does not rotate observations/actions or
change the torso or head pose.

## Lifecycle

| Event | Behavior |
| --- | --- |
| Process start | Register with Loop in IDLE; no arm service is opened yet. |
| Configure | Validate and save Node Config without opening hardware. |
| Start | Open hardware, or reuse it if hardware settings are unchanged; begin control ticks and observations. |
| Stop | Stop observation/control workers, discard pending arm/gripper commands, and hold the current arm position. Keep the hardware connection. |
| Fault | Pause both arms and report the failure to Loop. Reset Fault returns to IDLE; Configure then Start resumes operation. |
| Shutdown / process exit | Stop workers and close grippers and the shared Vega connection. |

The existing hardware initialization, including control-mode/head initialization,
happens on the first Start. After Stop, changing `frame_type`, the gripper, or
`action_frequency_hz` rebuilds the services on the next Start. Changing only
`control_frequency_hz` reuses the hardware connection. Every Start requires Configure.

## Data and commands

| Port | Data |
| --- | --- |
| `robot_observation` output | `left/right.cartesian_position`, gripper position, joints, velocities, torques, and wrench |
| `action_command` input | `left/right.target_cartesian_delta`: float64 `[6]`; `left/right.gripper_position`: scalar |
| `robot_command` request/reply | Home both arms through the existing reset paths; return failure if either arm fails. |

For an action, the Node reads both arms once and passes those pre-action states to
each arm’s Step. A failed Step faults the Node. Each common control tick:

1. Copies the latest measured state for both arms.
2. Sends both arms’ interpolated commands, if targets are available.
3. Passes the captured state to the observation worker for FK and publication.

The worker uses its own FK working data and keeps only the newest pending snapshot.
Slow encoding or publication does not block the control tick. Observations continue
before the first action arrives; their timestamp is the capture time. Sensor updates
remain asynchronous, and the publication rate can be lower than the tick rate when
processing cannot keep up.

Snapshots include the latest successful action’s diagnostics (`received_action`,
`left/right.action.*`). These persist until the next action and are cleared on Home
or Start; they are not an action/observation pair. Home suspends common control ticks
until both arms have completed the existing reset paths.
Loop’s Main Controller builds the final ControlStep for recording.
The action input keeps only the latest pending target.

## Development

```bash
./install.sh --extra robotiq --extra dev
.venv/bin/python -m pytest -q tests/loop_bridge
```

See [deployment](../../ansible/README.md). Legacy per-arm gRPC entrypoints remain
available for existing clients; the Loop deployment runs the single Robot Node.
