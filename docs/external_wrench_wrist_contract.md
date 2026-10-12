# Vega external wrench ownership

Vega RobotEnv publishes `external_wrench_sensor_raw`, its physical reference
point, and the two producer-owned frame rotations
`external_wrench_wrist_from_sensor_rotation` and
`external_wrench_world_from_sensor_rotation`. The source sample is never
labeled as a canonical wrist or world wrench.

RCI owns calibration only. After offset, gravity, force decoupling, torque gain,
and COM compensation, it applies the RobotEnv-provided rotations and publishes
the calibrated wrench as
`external_wrench_wrist` and `external_wrench_world` in the corresponding frames.
Both keep the physical F/T sensor measurement origin as their reference point.

The frame relationship is:

```text
calibrated_wrist = R_wrist_from_sensor @ calibrated_sensor
calibrated_world = R_world_from_sensor @ calibrated_sensor
```

For the validated Vega/RobotiQ hardware, the physical sensor axes are aligned
to Vega TCP and RobotEnv owns the company-wrist +165-degree profile. RCI does
not branch on the robot model or store that angle.

All fields use `wrench_on_robot`, `[Fx,Fy,Fz,Tx,Ty,Tz]`, N/Nm. If no accepted
calibration is active, RCI exposes raw plus frame metadata only and omits
canonical wrist/world.
