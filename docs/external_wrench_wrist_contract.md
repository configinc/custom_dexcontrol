# Vega external wrench ownership

Vega RobotEnv always publishes `external_wrench_sensor_raw`, its physical
reference point, and the two producer-owned frame rotations
`external_wrench_wrist_from_sensor_rotation` and
`external_wrench_world_from_sensor_rotation`. The source sample is never
labeled as a canonical wrist or world wrench.

RobotEnv owns calibration motion, fitting, cache, and application. After
offset, gravity, force decoupling, torque gain, and COM compensation, RobotEnv
publishes the calibrated wrench as
`external_wrench_wrist` and `external_wrench_world` in the corresponding frames.
The fit is performed at the physical F/T sensor origin. Published canonical
fields are referenced at the RobotiQ mounting origin, 18.5 mm along the
sensor/link-8 +Z axis. The corresponding moment translation (`r x F`) is
applied; Cartesian control and recorded pose remain referenced at `L_ee` or
`R_ee`.

The frame relationship is:

```text
calibrated_mount = shift(sensor_origin -> robotiq_mount, calibrated_sensor)
calibrated_wrist = R_wrist_from_sensor @ calibrated_mount
calibrated_world = R_world_from_sensor @ calibrated_mount
```

For the validated Vega/RobotiQ hardware, the physical sensor axes follow the
corresponding arm link-8 frame. Their fixed yaw relative to `L_ee`/`R_ee` is
owned by RobotEnv, as is the separate company-wrist +165-degree convention.
RCI does not store either transform; it only coordinates the two-arm stage
barrier and selects calibrated versus disabled inference input.

All fields use `wrench_on_robot`, `[Fx,Fy,Fz,Tx,Ty,Tz]`, N/Nm. If no accepted
calibration is installed, RobotEnv exposes raw plus frame metadata only and
omits canonical wrist/world. At the 07:00 KST expiry boundary RobotEnv removes
the installed applier before publishing the next observation, so canonical
wrist/world fields remain absent until a new calibration is committed.
