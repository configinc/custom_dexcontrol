# Managed Vega control architecture

This refactor separates transport, control calculations, output authorization,
and robot-specific stop behavior. The common Python lifecycle and safety policy
follow `robot_infra_project/robot-infra/projects/manipulation_controller`.
The implementation uses the existing Python SDK and arm algorithms.

## Responsibilities

| Module | Responsibility |
| --- | --- |
| `core/control/fsm.py` | IDLE / RUNNING / FAULT transitions and the first latched fault; no I/O |
| `core/control/safety.py` | Enabled checks, warn/block/fault policy and selected fault action |
| `core/control/motion_safety.py` | Collision/stability guard integration |
| `core/control/torque_limit.py` | Per-joint continuous arm position-offset relief from measured current |
| `core/control/collision.py` | Coal solid-shape clearance queries (segment checks are offline utilities) |
| `core/control/model_worker.py` / `motion_guard.py` | Shared collision/stability worker and nonblocking endpoint output selection |
| `core/control/stability.py` / `robot_model.py` | Static COM/support estimate and gripper assembly |
| `core/control/output.py` | Stage commands, reject revoked producers, validate the complete pending batch, publish |
| `core/control/actions.py` | One cancellable action worker and observable completion/error status |
| `robotenv_vega/control_node.py` | Vega lifecycle, arm/body modes, action admission and the output tick |
| `robotenv_vega/managed_hardware.py` | SDK composition, existing arm calculations, initial-pose action and hold behavior |
| `robotenv_vega/body_hardware.py` | Existing chassis and torso calculations, measured hold and stopped-state checks |
| `robotenv_vega/body_grpc.py` / `body_server.py` | Typed RPC adapters and process composition |

Normal output follows:

```text
RPC / action worker -> staged targets
                             |
200 Hz tick -> arm calculations / 50 Hz body calculations
                             |
             validate the complete pending batch
                             |
             arm current -> continuous position-offset correction
                             |
             read worker result; normal output or interpolate approved endpoint
                             |
                SDK publication + gripper dispatch
```

SDK components bound to the managed gate stage their local commands. A worker
cannot publish a delayed target after its ownership is revoked. SDK remote
motion-plugin targets are rejected in managed control because their intermediate
commands would bypass this tick. Direct standalone SDK use is unchanged.

Every pending normal component command is checked before any of that batch is
published. A failed right-arm calculation therefore discards the pending left-arm
and gripper commands as well. SDK writes are sequential, not a hardware-atomic
transaction: a transport failure during publication may follow an already sent
command. The node then latches FAULT and invokes hold. This is software control,
not a certified emergency-stop implementation or a replacement for hardware stop.

## Lifecycle and operations

| Request | Required state | Result |
| --- | --- | --- |
| Start | IDLE | RUNNING / hold; no initial-pose motion |
| SelectMode, EnableArms, Command, Step | RUNNING and the applicable mode | Existing teleop/body calculations |
| StartAction("initial_pose" / "fold") | RUNNING, no active action | ACTION mode; teleop blocked |
| Halt | Any | Cancel motion and hold; RUNNING remains authorized; FAULT remains latched |
| Stop | Any | Cancel motion and hold; RUNNING becomes IDLE; FAULT remains latched |
| ResetFault | FAULT, worker finished, valid stopped feedback | IDLE; no motion and no automatic Start |

Lifecycle and mode are separate. RUNNING can be in hold, arm handover, arms,
body, or action mode. Cancelling an action does not resume earlier teleoperation.
Completing an action returns to hold. A new arm-mode handover refreshes the fixed
torso model and reanchors Oculus before accepting arm commands.

`BodyState` reports `control_state`, `fault_reason`, `fault_action`, and the
action's `name`, `state`, and `error`. Both arm endpoints advertise
`body_control_lifecycle=true`. Use the matching RCI branch: main-page start paths
send Start, body/Quest input release sends Halt, and main-page stop sends Stop.
Robot Only starts a lifecycle session in hold. Vega Control exposes Initial Pose,
Fold, Cancel Action, and Reset Fault; it never starts control. Fault reset requires
the manual session to be stopped and leaves the server idle. Older managed
servers retain their old RCI behavior without the lifecycle capability.

## Actions

The named actions accepted by `StartAction` are `initial_pose` and `fold`.
Both require RUNNING and finish in hold; neither runs automatically at startup.

`fold` reads `poses.folded` and `actions.fold` from `vega_control.json`:

1. Fold the left arm, holding the right arm and torso at their current positions.
2. After measured left-arm completion, fold the right arm, holding the left arm and torso.
3. After measured right-arm completion, move the torso to `[0, 0, -1.5]` rad.
4. After measured torso completion, refresh fixed-torso IK and return to hold.

Each stage requires joint errors within 0.02 rad and measured speeds below
0.03 rad/s for 0.2 seconds before the next component can move.

The default joint reference speeds are 0.5 rad/s for arms and 0.2 rad/s for torso,
set by `actions.fold.arm_joint_velocity_rad_s` and `torso_joint_velocity_rad_s`.
SDK limits and the existing torso limit still apply. The producer uses measured path progress
with the body's 100 ms lead to avoid accumulating a distant target when tracking
is delayed. Each stage has a configurable 60-second timeout. Completion uses
feedback, not elapsed trajectory time alone. Stop/Halt cancels any stage;
a collision/stability rejection cancels to hold without continuing to the next
stage. The full arm/torso batch passes the ordinary output gate and safety checks.
The fold action keeps the current grasp and compensates the head for measured
torso pitch; it does not initialize or open the grippers.

While RUNNING, the common writer compensates the head reference for measured
torso pitch in arm/body/action modes and hold. The reference starts at the current
viewing direction; Initial Pose sets it to the configured head initial pose.
Head motion continues after the action completes, without a reach timeout or a
tracking-error fault. Joint limits and collision/stability approval still apply,
including when only the head moves in hold. Halt captures the current viewing
direction, and Stop/FAULT hold the measured head joints without compensation.

The folded arm angles preserve the prior viewer pose: left
`[1.57079, 0, 0, -3.071, 0, 0, -0.69813]`, right
`[-1.57079, 0, 0, -3.071, 0, 0, 0.69813]`. The vendor `folded` preset's -3.1 rad
elbows exceeded the URDF -3.071 rad limit and were previously clipped in the
viewer. The managed pose stores the valid values explicitly. A configured
out-of-range endpoint is rejected, not clipped when starting an action.

Call `BodyControl.StartAction` with `ActionRequest(name="fold")` after Start;
query `GetState.action` for completion or cancellation. No unfold action is
registered. The older standalone fold examples do not use this managed action.

`initial_pose` activates inactive Robotiq grippers explicitly, then performs
each arm's existing gripper-open / intermediate waypoint / initial-joint reset
sequence, followed by the configured head pose. It refreshes fixed torso IK
geometry afterward. Existing waypoint values, reset stepping and settling
tolerances are retained. A reset timeout is an error rather than successful
completion. The legacy per-arm Reset RPC maps to an exclusive single-arm action
with its existing home/target/random parameters; Reset(mode="current") remains
an observation-only request.

Action workers stage commands and check cancellation; the common tick owns
publication. Stop revokes output before waiting for a command lock. Robotiq Stop
discards queued targets and stops travel without requesting an open pose. A
command already in an I/O call finishes before the stop command can be sent.
Connecting or stopping an inactive gripper does not activate it.

Force calibration also executes through `ActionRunner`. Its existing per-arm
Start/Advance/Commit/Cancel/Status/Result RPCs remain the entry points; it is not
started with a bare `StartAction("force_calibration")` because the durable run
and fit inputs belong to that protocol. The first Start, admitted in RUNNING /
arms mode, creates the `force_calibration` action; the other arm joins it as a
participant. All stage and cancellation workers are registered with that action
and share its exclusive output owner. There is no separate calibration owner or
parallel controller-level action status.

The calibration manager retains the durable per-arm protocol, stage progress,
measurement, fitting, result persistence and bilateral commit. The action stays
running between stages and while awaiting Commit. One completed arm does not
release the other arm's control; after all participants finish and all workers
exit, the common action completion returns to hold. A participant failure faults
the action and cancels the other participant. Start, mode changes and new actions
cannot overtake cancellation cleanup.

The common 200 Hz tick dispatches buffered calibration interpolation through the
same output checks. During a calibration reset/return that sends joint commands
directly, interpolation is suspended only for that arm and its old buffer is
cleared. The other arm continues normally. Normal arm teleoperation retains its
existing calculations before the shared output safety stages.

Controller Stop/Halt and safety rejection cancel calibration without an automatic
return. An explicit per-arm Cancel RPC retains its existing safe-return option.
A completed READY_TO_APPLY candidate remains available for Commit after Stop or
restart; a late Commit changes calibration data without reclaiming robot output
or cancelling a subsequent action. Per-arm calibration phases/results remain on
the existing RPCs; `BodyState.action` consistently reports the common action's
name, running/cancelling/succeeded/cancelled/failed state and error.

## Configuration and fault action

Defaults are packaged in `src/dexcontrol/core/control/vega_control.json`.
Both `body_server` and `vega_pose_lab.py` accept
`--control-config /absolute/path/control.json` and read the same complete file.
Configuration is validated before hardware connection. It contains:

- `body_timeout_s`: 0.4 by default, preserving the body heartbeat interval.
- `collision`: collision margins, mode-specific exclusions, shared worker rates and geometry.
- `stability`: static support margin and approach/release distances.
- `model`: Robotiq mounting/mass/COM, fixed wheel support points and ground height.
- `poses.folded`: explicit managed arm/torso target angles shared with Pose Lab.
- `actions.fold`: arm/torso joint speeds, completion tolerances, settling interval and stage timeout.
- `arm_torque`: per-arm, per-joint torque estimates and continuous offset parameters.
- `input_timeout`, `execution_error`, `action_error`,
  `self_collision`, `static_stability`, and `arm_torque` policies.
- Each policy: `enabled`, `on_violation` (`warn`, `block` or `fault`), `fault_action`, and
  optional `log_interval_s` for warnings. Warn rules omit `fault_action`.

The supported Vega fault action is `hold`: zero chassis drive, capture measured
arm/torso/head positions, cancel queued targets and stop gripper travel. It does
not call software E-Stop, disable torque, open a grasp, or execute a return pose.
Execution/action errors must fault. Invalid/nonfinite output cannot be published
even when an optional policy is disabled. SDK joint-limit clamping remains in
place. Arm torque excess uses continuous position relief as described below;
there is no hard motor-current clamp or overcurrent E-Stop implementation.

### Continuous arm torque relief

The managed writer applies the same continuous-offset law as Torque Limit Lab's
`offset` method, independently to the seven joints of each arm. It uses total
current-derived torque, including gravity/load support, not external-contact
torque after gravity subtraction. An overload warns and corrects the target;
it does not by itself fault the controller.

For nominal arm target `q_user`, persistent offset `r`, estimated torque `T`,
limit `L`, gain `k`, recovery speed `v_recover`, relief speed `v_limit`:

```text
T = cur * torque_constant_nm_per_a * current_sign
recover_speed = min(v_recover, k * max(L + sign(r) * T, 0))
recovered = r - sign(r) * min(abs(r), recover_speed * dt)
relief_speed = -sign(T) * min(v_limit, k * max(abs(T) - L, 0))
r = recovered + relief_speed * dt
q_candidate = q_user + r
```

Recovery cannot cross zero. Recovery and relief are computed from the previous
offset and can add during torque reversal. There is no relief-active latch or
catch-up condition. When `r=0` and torque is within the limit, position commands
pass unchanged. New user target changes remain additive while an offset exists;
only the correction's speed is limited. Existing IK, arm interpolation and
velocity feedforward calculations are unchanged. The limiter forwards the
original velocity field, without adding an offset derivative or fading FF.

The only position clamp remains the SDK's existing absolute joint bounds.
Correction beyond those bounds is discarded so it cannot accumulate and delay
later recovery. No measured-position delta cap is introduced.

`safety.arm_torque.enabled` defaults to `false`; set it to `true` to enable the
feature. The torque feedback check and correction are inactive while disabled.
`on_violation` is `warn` when enabled.
Settings live in `arm_torque.arms.left_arm` and `right_arm`, with seven values
per field in SDK joint order, j1 through j7:

| Field | Packaged initial value | Unit |
| --- | --- | --- |
| `torque_constant_nm_per_a` | `[4.255, 4.255, 1.592, 1.592, 1.621, 1.621, 1.621]` | output N·m/A |
| `current_sign` | `[1, 1, 1, 1, 1, 1, 1]` | signed current-to-joint direction |
| `torque_limit_nm` | `[60, 60, 30, 30, 10, 10, 10]` | N·m |
| `gain_rad_per_nm_s` | `0.015` for each joint | rad/(N·m·s) |
| `max_relief_speed_rad_s` | `0.15` for each joint | rad/s |
| `recovery_speed_rad_s` | `0.15` for each joint | rad/s |

The torque constants already include the gearbox; do not multiply by gear ratio.
The limits are the supplied rated torques, used as initial tuning settings rather
than hardware-validated trip thresholds. The current type/scale and its sign in
joint coordinates must agree with those constants. `current_sign` explicitly
supports a reversed axis; its initial +1 setting has not been verified on hardware.

The limiter reads the SDK's cached arm state at the common output rate (normally
200 Hz), without issuing another remote request or waiting for a sample. It uses
`dt=1/control_hz`; delayed ticks do not cause a large elapsed-time catch-up step.
Sparse action targets are retained separately from corrected commands, so relief
continues between waypoints without adding the previous correction twice.

`arm_torque.feedback_timeout_s` defaults to 0.2 s and limits how long an unchanged
SDK state timestamp may be used after first consumption. It checks timestamp
progress using the local monotonic clock, not clock synchronization with the
robot; it does not establish the age of the first sample. Missing/nonfinite
current, invalid/backward timestamps, or stopped timestamp progress raise an
execution error before publishing the batch. Existing FAULT/hold then applies.
This feedback check is inactive when the limiter is disabled or holding.

Correction applies only to managed arm position commands in RUNNING arm/body/action
modes, including reset/fold/calibration targets. Torso, chassis, head and grippers
are not torque-limited. Stop, Halt, Fault, action handover/completion and mode
handover clear both the offset and its old nominal target while capturing the
existing measured hold. No relief motion is generated in IDLE, FAULT, hold, or
arm handover. Disabling the policy is a startup configuration choice; there is
no live toggle that drops an active offset.

Collision and stability approval run **after** torque correction. They can block
relief, and guarded endpoint interpolation can delay it. While model approval
holds output without an active approved target, offsets are not accumulated.
On a safety stop event, the old nominal target and correction are cleared along
with the producer buffers. Torque relief never bypasses a model hold. These
constraints and unknown drive dynamics mean this is an experimental software
relief controller, not a guarantee of peak torque or closed-loop stability.

`GetState.arm_torque` reports `disabled`, `idle`, `active`, or `model_hold`, and
per-arm joint names, currents, estimated torques, limits, nominal positions,
candidate positions/offsets, final approved output positions and over-limit flags.
Candidate offset values describe the candidate evaluated on that tick, even if
a model hold supersedes it. Final positions describe the admitted command, not
measured joint tracking. Existing clients may ignore this added protobuf field;
the RCI UI is unchanged.

Custom complete control JSON files must include both new `arm_torque` sections;
missing settings fail validation before connecting to hardware.

### Joint position bounds

Ordinary SDK position commands retain their existing joint-range clamp. The
managed controller adds no joint-range Fault policy on top of it. Model-approval
endpoints use the same SDK bounds so the collision check describes the target
that will be sent. A measured angle slightly beyond a bound does not cause Fault.

The safety worker uses raw measured angles for collision and stability geometry.
Its stationary hold targets are clipped to the SDK range, like ordinary SDK hold
commands, so feedback noise at a limit does not become an invalid new target.
There is no separate hard-coded joint-range fault inside collision approval.
Malformed/nonfinite data still follows `execution_error`. Endpoint margins and
worker-result expiry remain in `collision`/`stability`, and action completion
settings remain in `actions.fold`; they are not joint-range checks.

### Self-collision

The enabled `self_collision` policy uses `on_violation: "block"`. A proximity
block keeps the lifecycle RUNNING and retains arm/body teleoperation mode;
it does not latch a collision fault. Recovery uses the same input path.

| Setting | Default | Meaning |
| --- | --- | --- |
| `margin_m` | 0.01 | Minimum signed endpoint separation |
| `approach_m` | 0.03 | Enter guarded approval when measured minimum is below this |
| `release_m` | 0.04 | Return to ordinary output above this (hysteresis) |
| `monitor_hz` | 100 | Maximum worker request rate while control is active |
| `approval_hz` | 20 | Maximum candidate endpoint approval rate while guarded |
| `result_timeout_s` | 0.1 | Maximum age of the measurement used by a valid worker result |
| `max_outer_error_m` | 0.001 | Conservative outward simplification bound per shape |
| `collision_urdf`, `srdf` | omitted | Robot URDF and sibling SRDF; overrides use absolute paths |
| `excluded_links` | three wheel links | Upper-body collision coverage |

`model_worker.py` owns a separate model-only process. It receives named-joint
position snapshots as ordered arrays and has no SDK connection. In normal
operation it checks measured posture. Once near, it checks measured posture
and a candidate endpoint from the producer output. There is at most one
in-flight request; new input is coalesced, never queued behind old input.

`motion_guard.py` is the nonblocking part in the 200 Hz writer. Far from
both collision and tipping approach regions, ordinary arm interpolation and feedforward pass unchanged. On entry
it publishes a measured hold and clears buffered arm motion. Guarded motion
interpolates approved endpoints over a 50 ms interval; new approvals continue
from the last selected output. The last finite approved position is retained
until replaced. Arm velocity feedforward is zero in this guarded path; torso
velocity remains its existing actuator speed field. The 0.2 rad/s torso limit
remains in effect.

For each link pair, a target outside the margin must retain the margin. A pair
already inside the margin may stay inside only if its separation does not get
worse (2 µm numerical tolerance); a moving recovery target must improve at
least one offending pair. Improving the global minimum cannot hide worsening
another near pair. A neutral measured-position hold is permitted. Deferred gripper movement is
not treated as a neutral hold when the modeled robot is inside the margin.

A rejected endpoint replaces all staged output with measured holds and zero
base drive, discards deferred hand operations, stops gripper travel and clears
buffered arm motion. A rejected torso path is cancelled without overwriting the
operator's accepted goal; submit a new target to recover. Teleoperation retains
its mode. An action/calibration blocked at a waypoint is cancelled to hold,
rather than timing out into a collision fault. Stop, Halt and mode changes
revoke pending approvals. A worker reply authorizes only the captured command
batch, never a newer target. Missing/failed/stale worker results are explicit
`execution_error` faults; they do not silently disable checking.

Start and Reset Fault do not require already being outside the margin, so an
operator can supply recovery input. Other lifecycle and hardware checks remain.
Collision state, blocking reason, link pair, distance, required clearance,
measurement age and query time are exposed in `GetState.collision` independently
of lifecycle faults. The same model worker refreshes measured-only diagnostics
in IDLE, hold and FAULT. These read-only checks never approve a candidate or
release output. Ownership changes still discard old approvals; starting motion
requires a fresh active monitor result. The RCI Vega Control page displays
collision and static stability values, including their disabled/waiting states.

There is **no subdivision or swept/interpolation-path check in runtime**. The
endpoint policy does not guarantee the intermediate path, physical braking or
clearance during asynchronous motor tracking. `CoalCollisionModel.check_path`
remains an offline query utility only. The named fold action uses the same
monitor/endpoint approvals, without a separate collision bypass.

### Geometry and static stability

Pinocchio provides FK and Coal signed solid distances. Connected components use
conservative convex hulls; the 1 mm outer simplifier preserves containment.
Disconnected pieces stay separate. Assets, exclusions and hulls are loaded once
at startup. Every required joint needs finite feedback; model failures are errors.

Managed Robotiq arms get bundled 2F-85 full-stroke collision envelopes. The
external left/right grippers are mutually excluded only during manual `arms`
mode, through `collision.arms_mode_excluded_pairs` in `vega_control.json`.
All other modes, including actions, body and hold, include this pair. Changing
scope invalidates any pending approval before output resumes. Gripper-to-arm/body
checks remain in every mode. Pose Lab includes mutual gripper collisions.
Other gripper types do not acquire a Robotiq
model. The composed model retains the vendor robot joint kinematics and SRDF
exclusions. Envelopes avoid serial jaw reads in the
control path. See the asset README for nominal parallel-jaw coverage and sources.

The `model` section of `vega_control.json` sets tool mounting (left +90° / right −90°, sensor Z +18.5 mm),
tool mass/COM (existing calibration defaults: 1.2 kg, sensor [0,0,0.072] m),
fixed wheel support points and floor height. These are loaded by model assembly
and Pose Lab. The support triangle is fixed by design, ignoring steering/caster
changes. Defaults approximate the neutral URDF wheel contact centers:
rear [-0.299, 0.0015], right [0.185, -0.223], left [0.185, 0.223] m.

`stability.py` explicitly sums every URDF link mass, including the 88.43 kg fixed
base, and attached tool masses. It projects COM vertically onto level ground.
The minimum signed distance to the three support edge lines is positive inside,
negative outside. Pose Lab displays the result and geometry and exports them in
pose JSON. This is a static estimate without payload, acceleration or ground
slope. Mounting adapter geometry, cables and adaptive fingertip wrapping remain
outside collision coverage.

### Static tipping protection

The enabled `safety.static_stability` rule uses `on_violation: "block"`. Its
`stability` settings share `vega_control.json` with the rest of control:

| Setting | Default | Meaning |
| --- | --- | --- |
| `margin_m` | 0.02 | Minimum signed COM-to-support-edge margin |
| `approach_m` | 0.05 | Enter guarded output below 50 mm measured margin |
| `release_m` | 0.06 | Leave guarded output at or above 60 mm |

Collision and stability run in the **same model worker** on the same measured
state and candidate endpoint. The `collision` section's monitor/approval rates
and result timeout apply to both. No FK or COM query runs on the 200 Hz writer.
Either approach region enables 20 Hz approval; both checks must approve a
candidate. The checks can be enabled independently. Far from both regions,
existing interpolation and feedforward are preserved.

Each safe support edge must keep at least the configured margin. An edge
already below it may remain below only if it does not worsen (1 nm numerical
tolerance); recovery motion must improve at least one deficient edge. Improving
only the minimum margin cannot hide a different edge getting worse. A measured
hold is allowed. Below the margin, base drive is blocked even if the same batch
contains a recovery pose. Hand-only or unrelated motion is not recovery.

Rejection uses the same whole-batch hold, zero drive, buffer clearing and action
cancellation as collision. Ordinary arm/body mode stays active so the operator
can recover with a new input, without resetting a fault. Worker failures and
expired feedback still raise `execution_error`. `GetState.stability` reports the
current and last evaluated candidate margins (candidate absent on monitor-only
samples), required margin, COM, total mass, block reason, sample age and timing.

This is a **static, fixed-support guard**, not a dynamic tipping controller.
No payload, acceleration, slope, changing caster contact, braking distance or
intermediate-path guarantee is included. The 20/50/60 mm defaults are initial
configuration values, not physically validated limits. Pose Lab displays this
same configured margin but remains an offline editor: sliders are not blocked.

### Timing and placement

Use `examples/advanced_examples/benchmark_control_safety.py` for the current
worker/endpoint policy and a 200 Hz parent loop with mock feedback. It includes
1 mm geometry and both grippers. It reports worker computation, parent processing,
result age and scheduling lateness separately; SDK I/O and IK are excluded.

Local run on 2026-10-30 (500 random endpoints, 3 seconds per writer pose): static
COM/support query p95 0.060 ms; combined random-posture monitor p95 0.991 ms;
combined endpoint approval p95 1.172 ms. The 200 Hz writer's own processing p95
was 0.396 ms far / 0.405 ms near. At the folded pose the separate worker's mixed
monitor/approval queries reached p95 5.130 ms; this computation remains outside
the writer. These are local CPU measurements without robot I/O.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python \
  examples/advanced_examples/benchmark_control_safety.py \
  --iterations 2000 --writer-seconds 5 --output /tmp/vega-hybrid-collision-benchmark.json
```

Earlier synchronous full-path measurements in
[collision-hull-simplification.md](collision-hull-simplification.md) describe the
superseded implementation, not current writer cost. Endpoint geometry checks and
unit tests do not establish robot-PC deadlines or physical stopping distance.

Local sample on 2026-10-30 (i9-14900F, Python 3.12.13, pin 3.9.0, Coal 3.0.2,
2,000 random bounded configurations, seed 42; OPENBLAS/OMP threads = 1):

| Work | Median | p99 | Maximum |
| --- | ---: | ---: | ---: |
| Monitor FK + distances, tight CPU loop | 0.758 ms | 1.046 ms | 1.186 ms |
| Measured + target endpoint policy, tight CPU loop | 0.757 ms | 1.227 ms | 1.521 ms |
| Parent writer overhead, far (1,000 ticks) | 0.344 ms | 0.420 ms | 1.050 ms |
| Parent writer overhead, near (1,000 ticks) | 0.203 ms | 0.560 ms | 0.957 ms |

The separate near worker had p99 4.51 ms / max 5.43 ms, and sampled-result age
was at most 20.3 ms in this run. Tight-loop cache/warmup timing and periodically
scheduled process timing are distinct measurements. Both parent loops completed
without a timeout; these mock-I/O results do not claim robot-PC real-time
performance. Raw local report: `/tmp/vega-hybrid-collision-benchmark.json`.
