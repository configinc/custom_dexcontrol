# Vega deployment

Data Studio Layout's **Loop v2** mode installs under `~/loop-v2/`.
Deployment never stops or starts services. Stop Loop Robot/UTI before updating an
existing Loop installation.

- To use Loop: **Deploy Loop Nodes → Stop Existing Services → Start / Restart Robot/UTI**.
- To return: **Stop Robot/UTI**, then use the existing Robot, Inference and Recorder restart buttons.

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
| `dexcontrol_project_dir` | `/home/<dexmate_user>/loop-v2/custom_dexcontrol` on Vega |
| `dexcontrol_node_id` | `robot` |
| `dexcontrol_loop_endpoint` | Overrides `LOOP_NODE_GRAPH_NODE_ENDPOINT` in the launcher; empty inherits Vega's SSH environment |
| `dexcontrol_robot_name`, `dexcontrol_zenoh_config` | Optional DexComm settings, separate from the Loop connection |

The Loop endpoint must be reachable **from Vega**, which can have a different
network route from the Teleop gateway. The launcher registers in IDLE; Loop Start
opens the robot control resources. Gripper devices and control rates come from
[Node Config](../src/loop_bridge/README.md#node-config). Unit Config is not read.

`DOCKER_GITHUB_PAT` on the deployment server is passed transiently to the installer
for private SDK/Node downloads. Without it, Vega needs GitHub SSH access. An optional
`UV_DEFAULT_INDEX` is also forwarded when a private Python package index is used.
Neither is written into the startup script or Git configuration.

Deployment installs both Robotiq and SR drivers, serial permissions, and a dedicated
Python under `.python` with EtherCAT permissions. It does not change permissions
on a shared or system interpreter. Hardware selection happens at Configure.

Sync preserves `.venv`, `.python`, `third_party`, environment files, and `unit_config.json`.
A running `vega-loop-robot` session blocks deployment; stop Robot/UTI first.
The legacy `robot-server` session is separate. Deployment also saves the robot SSH
connection in `~/loop-v2/vega-connection.json` on the Teleop PC (owner access only),
so Service Controls can stop Vega without downloading a repository.

Restart the installed version without reinstalling:

```bash
ansible-playbook -i inventory.ini ansible/install/restart.yml
```

DS Layout's combined Robot/UTI control uses the installed startup scripts.
A running process can wait for Loop to become available; registration alone does
not activate robot control.
