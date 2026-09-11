# Vega deployment

Data Studio Layout's **Loop v2** mode installs under `~/loop-v2/`.
Deployment never stops or starts services. Stop Loop Robot/UTI before updating an
existing Loop installation.

- To use Loop: **Deploy Loop Nodes → Stop Existing Services → Start / Restart Robot/UTI**.
- To return: **Stop Robot/UTI**, then use the existing Robot, Inference and Recorder restart buttons.

The deployment server copies this checkout through the Teleop PC to the Vega PC.
Vega runs one dual-arm Loop Robot Node in a uv environment. The Teleop PC runs
a `socat` relay; UTI runs separately. Deployment installs both launchers without
starting either process.

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
| `dexcontrol_project_dir` | `/home/<dexmate_user>/loop-v2/custom_dexcontrol` on Vega |
| `dexcontrol_node_id` | `robot` |
| `dexcontrol_loop_endpoint` | GPU endpoint override; empty reads the Teleop PC's `LOOP_NODE_GRAPH_NODE_ENDPOINT` at start |
| `dexcontrol_relay_port` | `7448` on the Teleop PC |
| `dexcontrol_relay_bind_address` | Empty; detect the Teleop IPv4 address used to reach Vega |
| `dexcontrol_relay_tmux_session` | `vega-loop-relay` |
| `dexcontrol_robot_name`, `dexcontrol_zenoh_config` | Optional overrides for Vega's DexComm environment, separate from the Loop connection |

The Vega launcher loads `/etc/profile.d/10-dexmate-robot.sh` when present, just
as the legacy login shell did. This reads the device's existing `ROBOT_NAME`;
explicit deployment overrides are applied afterward.

Set `LOOP_NODE_GRAPH_NODE_ENDPOINT=tcp/GPU_HOST:7448` in the **Teleop PC's**
SSH environment. The relay accepts Vega connections on the Teleop interface
facing `dexmate_ip`, then forwards them to that GPU endpoint. TCP IPv4 addresses
and hostnames are supported. Use the GPU's wired address for the wired path.

```text
Vega Robot Node -> Teleop socat (:7448) -> GPU Loop
```

Deployment writes the relay address into Vega's startup script. Service start
redetects the address and passes it to the node, so it does not depend on a stale
tmux environment. The launcher registers in IDLE; Loop Start
opens the robot control resources. Arm pose presets, gripper devices, and control
rates come from [Node Config](../src/loop_bridge/README.md#node-config).
Unit Config is not read.

`DOCKER_GITHUB_PAT` on the deployment server is passed transiently to the installer
for private SDK/Node downloads. Without it, Vega needs GitHub SSH access. An optional
`UV_DEFAULT_INDEX` is also forwarded when a private Python package index is used.
Neither is written into the startup script or Git configuration.

Deployment installs both Robotiq and SR drivers, serial permissions, and a dedicated
Python under `.python` with EtherCAT permissions. It does not change permissions
on a shared or system interpreter. Hardware selection happens at Configure.

Sync preserves `.venv`, `.python`, `third_party`, environment files, and `unit_config.json`.
Running `vega-loop-robot` or `vega-loop-relay` sessions block deployment; stop Robot/UTI first.
The legacy `robot-server` session is separate. Deployment also saves the robot SSH
connection in `~/loop-v2/vega-connection.json` on the Teleop PC (owner access only),
so Service Controls can stop Vega without downloading a repository.

Restart the installed version without reinstalling:

```bash
ansible-playbook -i inventory.ini ansible/install/restart.yml
```

DS Layout calls `~/loop-v2/custom_dexcontrol/loop-service.sh` on the Teleop PC:

| Command | Behavior |
| --- | --- |
| `preflight COMMIT` | Check the installed robot revision, relay tools, and connection settings |
| `start` | Start the relay tmux, then the Vega robot tmux; remove the relay if robot startup fails |
| `stop` | Stop Vega, then the relay; attempt relay cleanup even if Vega is unreachable |
| `check-stopped` | Require both tmux sessions to be stopped |

The same commands can be run manually from the Teleop PC. Deployment saves paths
and session names in `.env.loop-service`; the robot SSH password remains in the
existing owner-only connection file. Start/stop implementation changes belong to
this repository; DS Layout only invokes the entrypoint.
A running process can wait for Loop to become available; registration alone does
not activate robot control.
