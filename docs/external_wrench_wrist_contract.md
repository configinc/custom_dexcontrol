# Vega external wrist wrench

Vega's F/T sensor output is source-native wrist data. It is not world-frame
data and must not be published as `external_wrench_world`.

The RobotEnv server publishes `external_wrench_wrist` only after applying the
configured active +Z yaw from sensor coordinates into the company wrist
convention. Force and torque use the same rotation and remain referenced at the
physical F/T sensor measurement origin.

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
values. A positive value means an active counter-clockwise rotation about +Z
when looking from the origin along +Z. Validate with independent +X and +Y
physical pushes before enabling data collection.
