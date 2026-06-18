<div align="center">
  <h1>🤖 Dexmate Robot Control and Sensing API</h1>
</div>

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)

## 📦 Installation

```shell
pip install dexcontrol
```

To run the examples in this repo, you can try:

```shell
pip install dexcontrol[example]
```

### Robotiq 2F-85 Gripper (optional)

If you want to use a Robotiq 2F-85 gripper connected via USB-RS485, clone this
repo with submodules and install the gripper package:

```shell
# Clone with submodule
git clone --recurse-submodules https://github.com/configint/custom_dexcontrol.git

# If you already cloned without --recurse-submodules:
git submodule update --init --recursive

# Install the gripper package
pip install -e robotiq_2f_85_controller/
```

Then start the RobotEnv server with `--gripper-type robotiq`:

```shell
python src/dexcontrol/core/robotenv_vega/server.py \
    --arm-side left \
    --gripper-type robotiq \
    --robotiq-comport /dev/ttyUSB0
```

Or instantiate `VegaRobot` directly:

```python
from dexcontrol.core.vega.robot import VegaRobot

robot = VegaRobot(
    arm_side="left",
    gripper_type="robotiq",
    robotiq_comport="/dev/ttyUSB0",
)
```

## Trajectory Interpolation & Control Tuning

`feature/interpolation-controller-upgrade` branch adds trajectory interpolation, output filtering, and runtime-tunable motor control parameters adapted from the official omniteleop `RobotController`.

### Architecture

```
gRPC Step() (~20Hz)              Background Control Loop (100Hz)
     |                                    |
     v                                    v
add_command_point()              execute_interpolated_tick()
     |                                    |
     v                                    v
TrajectoryInterpolator           interpolate(now) -> pos, vel
  (PCHIP cubic / linear)                  |
                                          v
                                  MultiChannelFilter
                                  (Butterworth / EMA)
                                          |
                                          v
                                  update_joints() -> hardware
                                  (delta clip, jerk limit,
                                   hw correction, vel feedforward)
```

### Launch Example (dual-arm with Robotiq)

```bash
# Left arm
python src/dexcontrol/core/robotenv_vega/server.py \
    --arm-side left \
    --grpc-port 50061 \
    --gripper-type robotiq \
    --robotiq-comport /dev/ttyUSB0 \
    --use-velocity-feedforward \
    --interpolation-method cubic \
    --interpolation-history 3 \
    --control-loop-hz 100 \
    --filter-type butterworth \
    --filter-cutoff-freq 15.0 \
    --vel-smoothing-alpha 0.15 \
    --hw-correction-alpha 0.4 \
    --max-jerk 0.15

# Right arm
python src/dexcontrol/core/robotenv_vega/server.py \
    --arm-side right \
    --grpc-port 50063 \
    --gripper-type robotiq \
    --robotiq-comport /dev/ttyUSB1 \
    --use-velocity-feedforward \
    --interpolation-method cubic \
    --interpolation-history 3 \
    --control-loop-hz 100 \
    --filter-type butterworth \
    --filter-cutoff-freq 15.0 \
    --vel-smoothing-alpha 0.15 \
    --hw-correction-alpha 0.4 \
    --max-jerk 0.15
```

### CLI Parameters

#### Interpolation

| Flag | Default | Description |
|---|---|---|
| `--interpolation-method` | `none` | `none`, `linear`, `cubic` (PCHIP). `cubic` requires >= 4 history points |
| `--interpolation-history` | `4` | Number of past input points for interpolation (min: 2) |
| `--control-loop-hz` | `0` | Background control loop rate. 0 = disabled (legacy sync path) |

#### Output Filter (applied after interpolation)

| Flag | Default | Description |
|---|---|---|
| `--filter-type` | `none` | `none`, `butterworth`, `ema` |
| `--filter-cutoff-freq` | `10.0` | Butterworth cutoff frequency (Hz). Higher = faster response |
| `--filter-order` | `2` | Butterworth filter order |
| `--filter-ema-alpha` | `0.1` | EMA alpha when `--filter-type=ema` |

#### Motor Control Tuning

| Flag | Default | Description |
|---|---|---|
| `--vel-smoothing-alpha` | `0.3` | Velocity feedforward EMA (0=smooth, 1=raw). Lower = smoother velocity |
| `--hw-correction-alpha` | `0.7` | HW feedback correction blend (0=ignore hw, 1=snap to hw). Lower = less position correction |
| `--max-delta-scale` | `1.0` | Scale factor for per-joint max delta clipping. >1 = allow larger steps |
| `--max-jerk` | `0.25` | Max acceleration change per step (rad/step^2). 0 = disable jerk limiting |

### Tuning Guide

| Goal | Key flags |
|---|---|
| Faster response | `--filter-cutoff-freq 20` `--interpolation-history 3` |
| Smoother motion | `--filter-cutoff-freq 5` `--vel-smoothing-alpha 0.1` `--max-jerk 0.15` |
| Less HW correction drift | `--hw-correction-alpha 0.3` |
| Disable filtering (interp only) | `--filter-type none` |
| Legacy mode (no interp) | Omit `--interpolation-method` and `--control-loop-hz` |

### Reference: Official omniteleop defaults (Vega YAML)

| Component | Filter | cutoff_freq | order |
|---|---|---|---|
| left_arm / right_arm | butterworth | 3.0 Hz | 2 |
| torso | butterworth | 5.0 Hz | 2 |
| left_hand / right_hand | butterworth | 8.0 Hz | 2 |
| chassis | butterworth | 5.0 Hz | 2 |

Note: The official config uses very conservative cutoff (3Hz for arms) designed for deployed production. For development/teleop, 8-15Hz is more practical.

---

## ⚠️ Version Compatibility

**Important:** `dexcontrol >= 0.5.0` requires robot firmware `>= 0.5.0`. Older firmware will not work with this release — update the firmware first, or pin `dexcontrol` to `0.4.x` until you can upgrade.

If your firmware is outdated, please update it before installing the new version to ensure full compatibility. Please contact the Dexmate team if you do not know how to do it.

### Firmware Compatibility Matrix

`dexcontrol` and the robot firmware are released in lockstep on the minor version: a given `dexcontrol 0.X.y` is compatible with firmware `0.X.*`. You can check the firmware version on your robot by running `examples/troubleshooting/display_robot_info.py`.

| `dexcontrol` version | Compatible firmware version |
| --- | --- |
| `0.5.x` | `0.5.x` |
| `0.4.x` | `0.4.x` |
| `0.3.x` | `0.3.x` |
| `0.2.x` | `0.2.x` |

If your firmware minor version does not match the `dexcontrol` version you intend to install, the client will print a version warning at startup and some features may not work correctly. Update the firmware first (see [vega-firmware](https://github.com/dexmate-ai/vega-firmware)) or pin `dexcontrol` to a version compatible with your current firmware.

**📋 See [CHANGELOG.md](./CHANGELOG.md) for detailed release notes and version history.**

## 📄 Licensing

This project is **dual-licensed**:

### 🔓 Open Source License
This software is available under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
See the [LICENSE](./LICENSE) file for details.

### 💼 Commercial License
For businesses that want to use this software in proprietary applications without the AGPL requirements, commercial licenses are available.

**📧 Contact us for commercial licensing:** contact@dexmate.ai

**Commercial licenses provide:**
- ✅ Right to use in closed-source applications
- ✅ No source code disclosure requirements
- ✅ Priority support options


## 📚 Examples

Explore our comprehensive examples in the `examples/` directory:

- 🎮 **Basic Control** - Simple movement and sensor reading
- 🎯 **Advanced Control** - Complex manipulation tasks
- 📺 **Teleoperation** - Remote control interfaces
- 🔧 **Troubleshooting** - Diagnostic and maintenance tools

---

<div align="center">
  <h3>🤝 Ready to build amazing robots?</h3>
  <p>
    <a href="mailto:contact@dexmate.ai">📧 Contact Us</a> •
    <a href="./examples/">📚 View Examples</a> •
  </p>
</div>
