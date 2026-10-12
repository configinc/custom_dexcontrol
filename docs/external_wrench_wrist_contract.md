# Vega external wrench ownership

Vega RobotEnv publishes only `external_wrench_sensor_raw` and its physical
reference point. It is source-native sensor data and is never labeled as a
canonical wrist or world wrench.

RCI owns calibration. After offset, gravity, force decoupling, torque gain, and
COM compensation, RCI publishes the same calibrated wrench as
`external_wrench_wrist` and `external_wrench_world` in the corresponding frames.
Both keep the physical F/T sensor measurement origin as their reference point.

The frame relationship is:

```text
calibrated_wrist = R_wrist_from_sensor @ calibrated_sensor
calibrated_world = R_world_from_wrist @ calibrated_wrist
```

All fields use `wrench_on_robot`, `[Fx,Fy,Fz,Tx,Ty,Tz]`, N/Nm. If no accepted
calibration is active, RCI exposes raw only and omits canonical wrist/world.
