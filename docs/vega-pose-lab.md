# Vega Pose Lab

An offline 3D pose editor for finding Vega arm/torso postures. It renders the
vendor visual URDF and evaluates the displayed configuration with the same
`CoalCollisionModel` used by managed control. It never creates a `Robot`,
opens SDK transport, reads live robot feedback, or sends motor commands.

## Run

In an environment with this checkout's dependencies installed:

```bash
python -m pip install -e '.[pose-lab]'
PYTHONPATH=src python examples/advanced_examples/vega_pose_lab.py \
  --host 0.0.0.0 --port 8081
```

Open `http://<this-computer-ip>:8081`. Viser serves its frontend locally; the
browser does not need a CDN. The viewer and server both load
`src/dexcontrol/core/control/vega_control.json`; use
`--control-config /absolute/path/control.json` to select another complete config.
The initial margin comes from `collision.margin_m`; mounting, mass and fixed
support geometry come from `model`. `--margin-mm` can override the display margin
for an experiment. UI margin changes never edit server safety settings.
Browser tabs share one pose in the running process. Restarting defaults to Folded; `--pose-json /path/to/vega-pose.json` restores an exported pose at startup.

## Use

- Start at **Folded**: SDK `folded` arm and torso values plus the `tucked` head.
- Use **Folded arms / upright torso** to inspect the previously reported arm
  overlap without folding the torso. Initial-arm and lifted-arm poses are also
  available. These are joint configurations, not executions of reset/fold actions
  or torso-compensated movement sequences.
- Select Torso, Left arm, Right arm or Head and drag a slider or type a joint
  value. All 20 controls use **radians** and the loaded URDF's joint bounds.
- Preset values outside URDF limits are clamped, with every adjustment shown.
  For dexmate-urdf 0.8.4 this includes `torso_j3` (-1.5708 to -1.5700 rad) and
  both `arm_j4` values (-3.1000 to -3.0710 rad). The viewer therefore explicitly
  distinguishes the bounded displayed pose from the raw SDK preset.
- Orbit by dragging, zoom with the wheel, and pan by right-dragging. **Reset
  camera** fits the current pose. Selecting a preset also fits that browser's camera.
- **OVERLAP** means at least one solid-shape pair clearance is negative. **BELOW
  MARGIN** means a checked pair is at or inside the selected margin. **CLEAR**
  means all checked pairs exceed it. The UI gives minimum clearance, overlap
  and violation counts, the ten closest link pairs, and computation time.
- Red collision shapes overlap; orange shapes are within the margin. Show all
  collision geometry, hide it, or hide the visual meshes to inspect the checked
  geometry. The closest-pair line joins Coal's nearest surface points.
- **Save / restore pose** contains **Download pose JSON**, a copyable/editable
  joint JSON field, and **Apply joint JSON**. Applying requires exactly the 20
  joint names with finite values inside limits. Invalid input leaves the pose
  unchanged and reports the error. Downloaded JSON includes named joint values,
  per-component arrays and all checked link-pair distances. To restore it, paste
  its `joint_positions` object into the field and click Apply.

## Model and stability

The viewer uses 1 mm conservative outer hulls and the same robot/SRDF assembly
as managed control. Both wrists carry Robotiq 2F-85 models, mounted at sensor
Z +18.5 mm with left +90° / right −90° yaw. The visual shows open jaws. Collision
envelopes cover the nominal parallel-jaw stroke, so collision checks are
independent of the displayed jaw position. The viewer includes external
left/right gripper collisions as well as gripper-to-arm/body pairs, matching
action/body/hold modes. Only manual Arms mode on the controller excludes the
pairs listed in `collision.arms_mode_excluded_pairs`.

The configured folded pose uses torso `[0, 0, -1.5]` rad and the existing arm
fold angles (elbows explicitly -3.071 rad within URDF limits). It is shared with
the managed `fold` action through `poses.folded` in `vega_control.json`. The
viewer uses a tucked head for this preset; the action retains the actual head pose.
Folded minimum clearance is approximately +20.46 mm, folded arms/upright torso
+20.46 mm, initial arms/upright torso +71.49 mm. A configuration check does not
validate the path or establish measured physical clearance.

The blue ground triangle uses three **fixed wheel support points**. Steering
and caster rotation do not move it. Yellow marks the whole-robot COM; its ground
projection is green above the configured safety margin, orange below that margin
inside the triangle, and red outside. Static tipping margin
is the minimum signed distance to a triangle edge line: positive inside,
negative outside. Total mass and COM coordinates update with joint sliders. The default folded
pose has about 136.3 mm static support margin with two Robotiq grippers.

Masses include the fixed base, all URDF inertials and 1.2 kg per gripper, with
COM at sensor [0,0,0.072] m from the force-calibration defaults. The nominal total
is 143.31 kg. Edit tool mounting/mass, fixed contact coordinates and floor Z in
the `model` section of `src/dexcontrol/core/control/vega_control.json`; restart to reload. Exported JSON
includes this configuration and the static stability result. The `stability`
section sets the runtime minimum margin (20 mm) and approach/release distances
(50/60 mm). `safety.static_stability` enables blocking on the managed server.
The viewer shows these settings while allowing arbitrary offline pose editing.

This is a level-ground, no-payload static estimate. It does not include motion
acceleration, slope, caster position or elastic deformation. The managed server
uses the same estimate to block unsafe endpoints near the support boundary;
this offline app does not command or stop a robot. Mounting adapter geometry, cables, payloads and adaptive
fingertip wrapping are not checked. Wheel meshes remain visible but are excluded
from self-collision checks. The app never connects to the physical robot.
