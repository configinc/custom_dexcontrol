# Robotiq 2F-85 model assets

Derived from [ROS-Industrial Robotiq](https://github.com/ros-industrial/robotiq/tree/45196f6558fe8ba9d89bc8a105396c68c3e7e892/robotiq_2f_85_gripper_visualization)
at commit `45196f6558fe8ba9d89bc8a105396c68c3e7e892` (BSD; see LICENSE).

- `open.glb`: merged upstream visual meshes at driver angle zero (open jaws).
- `*_envelope.obj`: base and two finger-group convex envelopes covering nominal
  parallel closing, driver angle 0–0.8 rad. Samples every 0.001 rad are expanded
  by a chain-length bound on point speed times half the step (approximately
  0.101 mm for moving groups), then outwardly expanded to cover 1 µm vertex
  quantization. They intentionally include empty space swept by the fingers.
- `geometry.json`: source revision, sampling step and motion padding.
- Adaptive fingertip wrapping around objects, cable routing, mounting adapter
  geometry and payloads are not represented. The visible open pose is not live
  jaw feedback. Runtime envelopes are further simplified with the configured
  bounded outer-hull tolerance (default 1 mm).

Rebuild with `examples/advanced_examples/build_robotiq_envelope.py`, passing an
expanded `robotiq_arg2f_85_model.xacro` URDF, the upstream visualization package
path and this output directory. Xacro and pycollada are build-time tools only;
no network, mesh download or xacro expansion is used at robot startup.

Mount and point-mass settings are in the `model` section of `../../vega_control.json`: 18.5 mm along the
physical link-8 sensor Z, left +90° / right −90° yaw (user-confirmed symmetric
90° mounting), 1.2 kg per tool and COM at sensor [0, 0, 0.072] m from the existing
force-calibration defaults. This is a static mass model, not rotational inertia
identification. Left/right external gripper collisions remain in the composed
model. The runtime excludes this pair only in manual Arms mode, as configured by
`collision.arms_mode_excluded_pairs`; other modes and Pose Lab include it.
Other robot link checks retain the vendor exclusions.
