# Vega external wrist wrench

Vega's F/T sensor output is source-native wrist data. It is not world-frame
data and must not be published as `external_wrench_world`.

The RobotEnv server publishes `external_wrench_wrist` only after applying the
configured active +Z yaw from sensor coordinates into the company wrist
convention. Force and torque use the same rotation and remain referenced at the
physical F/T sensor measurement origin.

The exact transform is:

```text
force_wrist  = Rz(yaw) @ force_sensor
torque_wrist = Rz(yaw) @ torque_sensor
```

Operational sign check: with `yaw = +90°`, sensor `+X` must be published as
wrist `+Y`. This is the definition to use when entering the measured value.
The company wrist axes must be identified physically on the robot before the
slot is populated; do not infer them from the sensor housing alone.

The signal uses `wrench_on_robot` sign, `[Fx,Fy,Fz,Tx,Ty,Tz]` ordering, N/Nm,
and is not bias- or gravity-compensated by this conversion.

Configure each arm independently:

```text
--external-wrench-sensor-to-wrist-yaw-degrees <measured-yaw>
```

The deployment slots are:

- `left_external_wrench_sensor_to_wrist_yaw_degrees`
- `right_external_wrench_sensor_to_wrist_yaw_degrees`

Both default to `null`. While unset, the server deliberately omits
`external_wrench_wrist`; it never relabels or zero-fills the raw sensor value.

## Calibration owner handoff

The Vega force-sensor owner must measure and populate the left and right yaw
values. Validate the result with independent sensor/wrist +X and +Y physical
pushes before enabling data collection.
