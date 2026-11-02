# Vega arm / body control

The managed server owns both arms, the chassis and the torso in one process.
The existing per-arm gRPC entrypoints remain available for legacy operation;
they do not provide the body-mode interlock. Do not run them alongside this server.

## Run

Use the robot's existing DexControl environment and communication configuration:

```bash
python -m dexcontrol.core.robotenv_vega.body_server \
  --host 0.0.0.0 --left-port 50061 --right-port 50063 --body-port 50065 \
  --gripper-type robotiq
```

Robotiq defaults to each arm's wrist RS485 pass-through, following the existing
per-arm server convention. For USB-RS485 adapters, add
`--left-gripper /dev/ttyUSB0 --right-gripper /dev/ttyUSB1` to the command.

This opens real hardware and performs the existing DexControl startup sequence:
head initialization, gripper opening, and each arm's middle/init pose movement.
After initialization it starts in **hold**, retaining the measured joint positions.
The supplied entrypoint supports built-in and Robotiq grippers. Both arms share one
SDK Robot. Body control runs at 50 Hz. Each arm uses the original independent
200 Hz output loop, with 20 Hz RCI input (`--action-hz`). An arm Step holds only
that arm's command-admission lock; IK does not lock either output loop.
Ports 50061/50063 must be free before startup.

Before opening the gRPC ports, the managed server keeps torso, head, arm,
chassis steering/drive, and configured wrist force sensor state subscriptions
`always_on` and waits for their initial data. State reception continues even
without observation requests, avoiding idle/resume gaps during concurrent
arm observations and force calibration. Missing startup data fails explicitly.
This changes state reception only; motion, Start/Stop, and calibration math
are unchanged. Camera and auxiliary sensor subscriptions retain their policies.

The default arm profile matches the existing 04 deployment: velocity feedforward
on, linear interpolation with 3 samples, velocity smoothing alpha 1.0, hardware
correction alpha 0.5, max delta scale 3.0, max jerk 0 (disabled), rotation
sensitivity 2.0 and velocity ratio 1.0. These are CLI options on body_server.
The per-arm server's defaults and command calculations remain unchanged.
This branch is based on `refactor/vega-force-calibration-clean-v2`, preserving
its force-calibration RPCs, wrench observations, and torso/head observation fields.

RCI must use both managed gRPC endpoints. Its connection code recognizes
`body_control_required=true` and skips automatic homing. Open RCI's
`/vega-control` page after connecting the arms and internal Oculus controller
and starting Robot Control on the main RCI page. Only the main page starts or
stops the control session; Vega Control switches modes within that session.
Stopping Robot Control sends body Stop as well as stopping arm output. Mode
and motion requests are rejected while the main control session is stopped.
The body gRPC endpoint uses the robot IP and port advertised by both arms
(default 50065). There is no separate endpoint override or server identity
handshake; connect both arms to the intended robot as with existing RCI control.

## Deploy from data-studio-layout

Choose **Custom Dexcontrol**, select this branch or its release tag, and deploy
to the Vega teleop unit using normal deployment mode (not Loop node installation).
DS Layout runs this revision's `ansible/install/deploy.yml`: it syncs through the
teleop PC to the robot PC, installs dependencies, then runs `restart.yml`.

For a Robotiq unit, restart now launches **one** `body_server` in the existing
`robot-server` tmux session. The arm ports stay 50061 and 50063; body gRPC uses
50065. It removes old per-arm and managed server processes before starting and
checks all three ports. Normal startup still opens the grippers and moves the
arms through their existing initialization poses. SIGTERM/SIGHUP run stop/hold
cleanup. The tested arm profile remains 20 Hz input / 200 Hz output.

The managed server starts in a Bash login shell to load the robot's
`ROBOT_NAME` and communication environment. Starting with `bash -ic` alone
can leave the robot namespace empty and fail hand-type service discovery.

Managed-server startup output is retained in the `robot-server` tmux pane and
`~/.local/state/dexcontrol/body_server.log` on the robot PC, including when
startup exits before the gRPC ports open. The log is outside the synced checkout
so a subsequent deployment does not delete it. Inspect it with
`tail -n 100 ~/.local/state/dexcontrol/body_server.log` or
`tmux capture-pane -pt robot-server -S -100` when a port readiness check fails.

Gripper type is read from the unit configuration. Both `robotiq` and
`robotiq_vega` start the managed server with the same Robotiq driver, using
`robotiq_transport=rs485` by default. This selects each arm's wrist RS485
pass-through without probing ttyUSB devices or passing serial-port arguments.
Use `-e robotiq_transport=usb` for USB-RS485 adapters. USB mode retains the
existing sorted ttyUSB mapping, with `left_gripper_port` and
`right_gripper_port` available as overrides. There is no automatic switch
between transports.

From an updated SDK checkout on the teleop PC, after syncing the code to the
robot, restart using unit 04's USB mapping with:

```bash
ansible-playbook -i localhost, -c local ansible/install/restart.yml --ask-become-pass \
  -e gripper_type=robotiq -e robotiq_transport=usb \
  -e left_gripper_port=/dev/ttyUSB0 -e right_gripper_port=/dev/ttyUSB1
```

The same extra variables apply to a full `ansible/install/deploy.yml` deployment.
`vega_body_port` overrides the body port. `vega_body_control=false` selects
legacy per-arm servers with the same transport setting. SR, Wuji and Vega Hand
deployments keep their existing per-arm launch path; this release's managed
Ansible deployment supports Robotiq only.

Then deploy the matching `robot-control-interface` revision. Its existing web
process discovers the body endpoint from both arm services. No DS Layout source
change or separate Oculus service is required.

When replacing the temporary unit-04 test install, first stop
`temp-sds-rci.service` on the teleop PC and `temp-sds-body.service` on the robot
PC. Normal deployment uses `~/robot-control-interface`, `~/custom_dexcontrol`
and tmux; it does not manage the temporary test units. Keep them stopped when
using the normal deployment.

The Robotiq submodule is pinned to the same revision as the restored
`v3.1.0-wrench-beta-3` server, including USB and wrist RS485 support. DS Layout
initializes the pinned submodules before Ansible syncs them.

## Modes and handover

- **body**: clear pending arm targets; hold measured arm joints and retain grasp.
  Accept periodic chassis velocity and absolute shoulder goals.
- **prepare_arms**: clear buffered commands, stop the body, and require
  post-stop sensor updates with velocities below 0.01 for at least 150 ms.
  Rebuild both reduced arm IK models from one measured arm/torso snapshot,
  keeping the torso joints locked at those angles.
  Return `arm_ready`, without enabling arm motion.
- RCI fetches fresh observations for both arms, updates only Oculus Cartesian
  anchors (preserving gripper target and VR axes), then enables arm control.
- **arms**: accept normal Step/Reset commands. The body stays stopped.
- **hold**: zero chassis drive, hold current torso/arm positions, discard queued
  targets. Motion requests are rejected while held.

Body input expires after 0.4 seconds. Body faults and expired body input enter
hold and require explicit mode selection. Arm commands retain the original
per-arm error handling and have no added body watchdog or bilateral fault stop.
An explicit Stop or body-mode transition closes arm command admission, stops
the independent output loops, drains in-flight Steps and clears buffered targets
before holding the measured pose. The body loop does not run additional continuous freshness or
`is_active()` checks. The UI sends body commands
at 10 Hz, only from the window that selected body mode.

RCI mode transitions are limited to manual teleop: stop policy inference and
calibration before changing mode. The body feature is not a new policy action
space and is not included in existing arm training/recording vectors.

Main-page calibration prepares the managed arm command path without requiring
Oculus tracking, then returns to hold when it finishes or is cancelled. Server
force-calibration motion is admitted only in arm mode. Stop cancels active
calibration without a return movement and stops any reset already in progress.
The wrench server's durable `READY_TO_APPLY` candidate is already motionless and
remains available for a later commit. Reset/calibration completion cannot restart
the arm output loop after a body Stop.

The loop schedules against a monotonic deadline rather than adding a fixed
sleep after each tick. Body targets integrate the actual elapsed time; a new
velocity applies only from its arrival. Mode handover still confirms a stopped
pose before reanchoring. The original deployed arm limits and tuning are preserved.

## Shoulder control and limits

Body commands carry base velocity `[vx, vy, wz]` (m/s, m/s, rad/s) and an
optional absolute `shoulder_target` `[x, z, pitch]` (m, m, rad), all in the robot
base frame. Omitting the shoulder target retains the accepted goal.

The controlled point is `arm_center`, the center between arm mounting frames.
X is forward; Z is up. Pitch is rotation about base +Y: 0 means the shoulder
frame's +Z is upright; positive tilts it forward. This differs from SDK
`Torso.pitch_angle`, which measures from the horizontal plane.

The RCI web app accumulates button jog input into an absolute shoulder goal;
it can also send X/Z/pitch entered directly. DexControl never integrates a
shoulder velocity request. It validates the final goal and an approximately
straight Cartesian path before acknowledging the command. Waypoints are at most
1 mm / 0.002 rad apart; analytic IK resolves each waypoint using the previous
joint pose as the reference. Every waypoint must respect the native URDF/SDK
joint limits and singularity check. Invalid goals or paths return INVALID_ARGUMENT
without changing the accepted goal, base velocity, or active trajectory.

Joint-linear interpolation connects the waypoints. Each segment's duration is
`max(abs(q_next - q_start) / joint_velocity_limits)`, so all intermediate joint
commands remain inside the limits and each joint stays at or below 0.2 rad/s
(or a lower native limit). The body loop projects measured joints onto the path
and allows a 0.1-second reference lead beyond that physical progress, advancing
it by at most one elapsed control interval per tick. This lead addresses observed
torso startup deadband: isolated J2 goals of 0.004/0.008 rad did not move, while
0.016 rad did. Slow hardware therefore slows path progress instead of letting
wall-clock time run the reference to a distant final goal. There is no tracking
error cutoff, target overwrite or automatic recovery.
Retargeting starts at the current commanded pose, without a position jump.
Identical target packets keep the existing trajectory and progress.

The SDK torso velocity argument is a positive movement speed, not the signed
velocity feedforward used by arms. While executing an accepted target, each
torso joint receives its speed cap (at most 0.2 rad/s), including after the
planned path time ends. The final position remains commanded so delayed joints
can finish approaching it. Explicit hold uses the measured position and zero
speed. Trajectory timing alone does not establish that the physical torso has
reached the target.

State `shoulder_target` and `torso_target` are the accepted final goals, not the
intermediate commands or measured feedback. Releasing a button freezes the final
goal and execution continues to it. Stop, mode changes and body watchdog expiry
cancel the path and reanchor the goal at measured joints. The UI continues sending
base-zero/target keepalives while a trajectory executes.

The locally inspected `vega_1p_gripper.urdf` has:

| Joint | Lower (rad) | Upper (rad) |
| --- | ---: | ---: |
| torso_j1 | 0 | 1.570 |
| torso_j2 | 0 | 3.141 |
| torso_j3 | -1.570 | 1.570 |

The server uses runtime values, not this table. Command caps are base X/Y
0.15 m/s, yaw 0.3 rad/s, shoulder X/Z 0.02 m/s and pitch 0.05 rad/s;
torso joints are capped at the minimum of URDF, SDK and 0.2 rad/s.
There is no additional fixed target-tracking error cutoff. Joint speeds are
bounded in the trajectory; measured lag does not redefine the accepted goal.
These are kinematic paths, not collision checking.

Arm IK optimizes only the 14 arm joints, as in the original per-arm server.
The torso is fixed model geometry, never an optimization variable. Before
returning to arm mode, the managed server rebuilds both reduced models with
the stopped, measured torso angles. Fresh arm observations and Oculus anchors
then use that geometry. Model refresh sends no hardware commands and performs
no startup/reset movement. A failed refresh keeps control in hold.
During body mode, arm observations transform the fixed-model arm pose through
the measured shoulder pose, so they still report the actual base-frame pose.
This observation-only FK transform neither runs IK nor commands the arms.
The original per-arm entrypoint retains its existing fixed-torso model behavior.

## Steering

Runtime steering limits are the intersection of URDF joint limits and
`max_steering_angle`. The locally inspected config uses ±2.35 rad (±134.65°),
which is narrower than the inspected URDF's ±2.722 rad.

The wheel solver filters out infeasible steering angles **before** choosing the
nearest forward/reverse drive solution. Final `set_motion_state` also enforces
the limits. Non-finite commands are rejected. A direction change brakes first,
then steers with drive held at zero, then accelerates when both errors are at
most 0.05 rad. Stop retains the current steering angle.

The managed loop sends nonblocking commands every tick and has an explicit stop;
it does not use `wait_time`. The separate SDK timed-command API was fixed to
use the full requested duration, without subtracting one second. That legacy
API still does not imply an automatic final stop.

## BodyControl gRPC API

All three endpoints run in one process and share one SDK Robot: left arm 50061,
right arm 50063, body 50065. The body service is `vega.body.v1.BodyControl`,
defined in `proto/vega_body.proto`. The robot no longer hosts a body HTTP server.
The browser-to-RCI web API remains HTTP.

Both arms advertise `body_control_required=true` and `body_control_port` through
the existing GetConfig RPC. RCI uses those capabilities, the same robot IP and
matching body ports to enable the UI and connect. GetState returns body state, including measured and applied target shoulder/joint values;
arm observations continue to use the existing RobotEnv service.

RCI is the single control source and serializes mode changes. The server executes
commands according to its current mode, velocity/joint limits and input timeout.
There is no command-session token or sequence counter. A delayed command arriving
after returning to the same mode is treated as a current command. This is an
accepted simplification for this teleop setup, rather than a multi-controller
arbitration protocol.

RCI suspends arm sending before a mode change. Returning to arms still follows
stop → fresh observations → Oculus reanchor → EnableArms. Stop interrupts a pending
transition; arm enable is accepted only while ARM_READY. Ordinary arm RPC payloads
and metadata remain unchanged.

| RPC | Request / behavior |
| --- | --- |
| GetState | empty request; returns typed mode, limits, shoulder, joints and wheels |
| SelectMode | BODY or ARMS; ARMS prepares ARM_READY |
| EnableArms | empty request, after RCI reanchors |
| Command | base Velocity plus optional absolute ShoulderPose; returns accepted state |
| Stop | empty request; stop and clear buffered targets |

`BodyCommand.velocity` uses `base_vx`, `base_vy`, and `base_wz`.
`BodyCommand.shoulder_target` uses `x`, `z`, and `pitch`. The old shoulder velocity
fields are retained only as jog-speed metadata in Limits; nonzero torso velocity
in a command is explicitly rejected. Non-finite values, invalid goals/paths and
exceeded base velocity limits return INVALID_ARGUMENT. Wrong mode returns
FAILED_PRECONDITION. RCI uses a 5-second RPC deadline and the independent robot-side
0.4-second body watchdog remains active. Stop can interrupt a pending mode transition.

The two repositories carry identical proto definitions and generated bindings.
Generate from the repository root with `bash proto/compile_proto.sh`, using
Python 3.10, grpcio-tools 1.48.2, protobuf 3.20.1 and setuptools <81 to retain
compatibility with RCI's existing robot integrations. Do not upgrade the shared
protobuf dependency just to regenerate this service.

## Validation

Run the hardware-free mode, kinematics and adapter tests with NumPy, pytest, grpcio and protobuf:

```bash
python -m pytest tests/test_body_control.py tests/test_body_hardware.py tests/test_torso_kinematics.py tests/test_body_grpc.py
```

Physical robot startup, tracking, braking and grasp retention require a separate
hardware test. The development verification did not actuate a robot.
