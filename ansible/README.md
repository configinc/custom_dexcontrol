# Vega deployment

The deployment server copies this checkout through the Teleop PC to the Vega PC.
Vega runs one dual-arm Loop Robot Node in a uv environment. UTI runs separately.

```bash
ansible-playbook -i inventory.ini ansible/install/deploy.yml --ask-become-pass \
  -e dexcontrol_loop_endpoint=tcp/loop-host:7448
```

The checkout must include its gripper submodules (`git submodule update --init`).
Host setup requires sudo on the gateway and passwordless sudo on Vega, as in the
existing deployment. Override the Vega connection settings in inventory.

| Setting | Default / purpose |
| --- | --- |
| `dexmate_ip`, `dexmate_user`, `dexmate_pass` | Existing Vega connection settings; traffic passes through the Teleop PC |
| `dexcontrol_python` | `3.12`; the pinned OMPL dependency has no Python 3.13 wheel |
| `dexcontrol_project_dir` | `/home/<dexmate_user>/custom_dexcontrol` on Vega |
| `dexcontrol_start` | `true`; set `false` to install and configure without starting |
| `dexcontrol_node_id` | `robot` |
| `dexcontrol_loop_endpoint` | Overrides `LOOP_NODE_GRAPH_NODE_ENDPOINT` in the launcher; empty inherits Vega's SSH environment |
| `dexcontrol_robot_name`, `dexcontrol_zenoh_config` | Optional DexComm settings, separate from the Loop connection |
| `gripper_type` | `robotiq`, `sr_gripper`, or `default` |
| `dexcontrol_left_gripper_port`, `dexcontrol_right_gripper_port` | Prefer explicit serial paths or EtherCAT interfaces; otherwise the launcher discovers exactly two devices |
| `dexcontrol_robot_args` | Existing controller tuning, including linear interpolation and the 200 Hz control loop |

The Loop endpoint must be reachable **from Vega**, which can have a different
network route from the Teleop gateway. The launcher registers in IDLE; Loop Start
opens the robot control resources. Gripper discovery belongs to launcher setup.

`DOCKER_GITHUB_PAT` on the deployment server is passed transiently to the installer
for private SDK/Node downloads. Without it, Vega needs GitHub SSH access. An optional
`UV_DEFAULT_INDEX` is also forwarded when a private Python package index is used.
Neither is written into the startup script or Git configuration.

SR deployment installs a dedicated Python under `.python` for EtherCAT permissions;
it does not change permissions on a shared or system interpreter.

Sync preserves `.venv`, `.python`, `third_party`, environment files, and `unit_config.json`.
It stops the managed `robot-server` tmux session before updating source. Legacy
left/right server panes receive Ctrl+C and are removed after their processes exit.
The new process uses the same session name.

Restart the installed version without reinstalling:

```bash
ansible-playbook -i inventory.ini ansible/install/restart.yml
```

DS Layout's Vega Robot Restart already calls this playbook. It checks the Node
process rather than the obsolete gRPC ports 50061/50063. A running process can wait
for Loop to become available; registration alone does not activate robot control.
