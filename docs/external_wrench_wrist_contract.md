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
fields use the configured TCP as their anchor convention, with rotation only
and no moment translation (`r x F`).

The frame relationship is:

```text
calibrated_wrist = R_wrist_from_sensor @ calibrated_sensor
calibrated_world = R_world_from_sensor @ calibrated_sensor
```

For the validated Vega/RobotiQ hardware, the physical sensor axes are aligned
to Vega TCP and RobotEnv owns the company-wrist +165-degree profile. RCI does
not branch on the robot model or store that angle; it only coordinates the two
arm stage barrier and selects calibrated versus disabled inference input.

All fields use `wrench_on_robot`, `[Fx,Fy,Fz,Tx,Ty,Tz]`, N/Nm. If no accepted
calibration is installed, RobotEnv exposes raw plus frame metadata only and
omits canonical wrist/world. Cache validity is checked when an inference run is
selected: crossing the 07:00 KST boundary does not mutate an already-running
inference session, but the next run must calibrate again.
