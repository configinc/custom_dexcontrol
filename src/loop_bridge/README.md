# Vega Robot Node

One external `loop.robot@1` Node controls both arms of a Vega through the existing
DexControl IK, interpolation, filtering, and gripper implementation.

## Install and run

From the repository root, with Git, uv, a C++ compiler, CMake, and GitHub access:

```bash
./install.sh --extra robotiq
export LOOP_NODE_GRAPH_NODE_ENDPOINT=tcp/loop-host:7448
./run.sh --node-id robot
```

The installer creates a Python 3.12 `.venv`, downloads pinned SDK/Node sources to
`third_party`, and installs the selected gripper. Use `--extra sr-gripper` for SR
EtherCAT grippers, or omit the gripper extra for built-in hands. With SR, the
per-arm Node Config devices are network interface names. Ansible configures the
required EtherCAT permissions.

`--loop-endpoint` overrides `LOOP_NODE_GRAPH_NODE_ENDPOINT`; the fallback is
`tcp/127.0.0.1:7448`. This is the Loop connection. DexComm's `ROBOT_NAME` and
`ZENOH_CONFIG` still configure communication with the physical Vega.

Match `--node-id` to the Robot Node ID in the Loop Cell Config.

## Node Config

```json
{
  "control_hz": 20,
  "observation_frequency_hz": 20.0,
  "gripper_type": "robotiq",
  "left_gripper_device": "/dev/ttyUSB0",
  "right_gripper_device": "/dev/ttyUSB1"
}
```

| Field | Meaning |
| --- | --- |
| `control_hz` | Main Controller’s action frequency, used by the arm controller |
| `observation_frequency_hz` | Regular state publication rate, independent of action arrivals |
| `gripper_type` | `default` (built-in), `robotiq`, or `sr_gripper` |
| `left_gripper_device`, `right_gripper_device` | Distinct serial paths for Robotiq or network interfaces for SR; ignored for built-in grippers |

For SR, set both devices to EtherCAT interfaces such as `enp1s0` and `enp2s0`.
The selected driver must be installed; Ansible installs both optional drivers.
The existing deployed IK and interpolation tuning remain internal defaults,
including the 200 Hz control loop. Actions use `target_cartesian_delta` and
normalized gripper `position`. No DS Layout Unit Config lookup is required.

## Lifecycle

| Event | Behavior |
| --- | --- |
| Process start | Register with Loop in IDLE; no arm service is opened yet. |
| Configure | Validate and save Node Config without opening hardware. |
| Start | Open hardware, or reuse it if hardware settings are unchanged; resume control and publish an initial observation. |
| Stop | Stop observation/control workers, discard pending arm/gripper commands, and hold the current arm position. Keep the hardware connection. |
| Fault | Pause both arms and report the failure to Loop. Reset Fault returns to IDLE; Configure then Start resumes operation. |
| Shutdown / process exit | Stop workers and close grippers and the shared Vega connection. |

The existing hardware initialization, including control-mode/head initialization,
happens on the first Start. After Stop, changing the gripper or `control_hz`
rebuilds the services on the next Start. Changing only the observation rate reuses
the hardware connection. Every Start requires Configure.

## Data and commands

| Port | Data |
| --- | --- |
| `robot_observation` output | `left/right.cartesian_position`, gripper position, joints, velocities, torques, and wrench |
| `action_command` input | `left/right.target_cartesian_delta`: float64 `[6]`; `left/right.gripper_position`: scalar |
| `robot_command` request/reply | Home both arms through the existing reset paths; return failure if either arm fails. |

For an action, the Node reads both arms once and passes those pre-action states to
each arm’s Step. A failed Step faults the Node. Observations are published on their
own timer with current state and the latest successful action’s diagnostics
(`received_action`, `left/right.action.*`). These diagnostics persist until the next
action and are cleared on Home or Start; they are not an action/observation pair.
Loop’s Main Controller builds the final ControlStep for recording.
The action input keeps only the latest pending target.

## Development

```bash
./install.sh --extra robotiq --extra dev
.venv/bin/python -m pytest -q tests/loop_bridge
```

See [deployment](../../ansible/README.md). Legacy per-arm gRPC entrypoints remain
available for existing clients; the Loop deployment runs the single Robot Node.
