# Vega deployment

Data Studio Layout's **Loop v2** mode installs under `~/loop-v2/`.
Deployment never stops or starts services. Stop Loop Robot/UTI before updating an
existing Loop installation.

- To use Loop: **Deploy Loop Nodes → Stop Existing Services → Start / Restart Robot/UTI**.
- To return: **Stop Robot/UTI**, then use the existing Robot, Inference and Recorder restart buttons.

The deployment server copies this checkout through the Teleop PC to the Vega PC.
Vega runs one dual-arm Loop Robot Node in a uv environment and connects directly
to Loop on the Teleop PC. UTI runs on Teleop separately. Deployment installs the
Vega launcher without starting the Robot Node.

```bash
ansible-playbook -i inventory.ini ansible/install/deploy.yml --ask-become-pass
```

The checkout must include its gripper submodules (`git submodule update --init`).
Host setup requires sudo on the gateway and passwordless sudo on Vega, as in the
existing deployment. Override the Vega connection settings in inventory.

| Setting | Default / purpose |
| --- | --- |
| `dexmate_ip`, `dexmate_user`, `dexmate_pass` | Existing Vega connection settings; traffic passes through the Teleop PC |
| `dexcontrol_python` | `3.12`; the pinned OMPL dependency has no Python 3.13 wheel |
| `dexcontrol_install_timeout_seconds` | `900`; maximum duration of each Vega installation stage and its temporary proxy |
| `dexcontrol_project_dir` | `/home/<dexmate_user>/loop-v2/custom_dexcontrol` on Vega |
| `dexcontrol_node_id` | `robot` |
| `dexcontrol_loop_endpoint` | Direct Loop endpoint override; empty reads Teleop's `LOOP_NODE_GRAPH_NODE_ENDPOINT` at start |
| `dexcontrol_robot_name`, `dexcontrol_zenoh_config` | Optional overrides for Vega's DexComm environment, separate from the Loop connection |

The Vega launcher loads `/etc/profile.d/10-dexmate-robot.sh` when present, just
as the legacy login shell did. This reads the device's existing `ROBOT_NAME`;
explicit deployment overrides are applied afterward.

Run Loop on the **Teleop PC**, listening on its Vega-facing interface. Loop's
default `tcp/0.0.0.0:7448` listener supports this. Set the direct endpoint manually
in Teleop's SSH environment, replacing any previous GPU address:

```bash
export LOOP_NODE_GRAPH_NODE_ENDPOINT=tcp/TELEOP_VEGA_IP:7448
```

Use Teleop's wired IP reachable from Vega, not `127.0.0.1` or `0.0.0.0`.
UTI on Teleop can use the same endpoint. Alternatively, set
`-e dexcontrol_loop_endpoint=tcp/TELEOP_VEGA_IP:7448` when deploying this robot.
The script passes the configured address directly to Vega; it does not detect
addresses or create a runtime TCP relay.

```text
Vega Robot Node -> Teleop Loop (:7448)
```

Service start passes the configured endpoint to the node explicitly, so it does
not depend on a stale Vega tmux environment. The launcher registers in IDLE; Loop Start
opens the robot control resources. Arm pose presets, gripper devices, and control
rates come from [Node Config](../src/loop_bridge/README.md#node-config).
Unit Config is not read.

Vega does not need its own internet connection for deployment. Each installation
stage starts a temporary Tinyproxy process on the Teleop PC and routes Vega's
APT, pip, uv, and Git HTTPS downloads through it. Teleop must have internet access
and be able to reach Vega over SSH. The proxy binds only to Teleop's Vega-facing
IPv4 address, on an automatically allocated port, and accepts only Vega's IPv4
address. DNS for downloads is resolved on Teleop. No route, DNS, firewall, or
persistent proxy environment is changed on either host. The gateway installs
`tinyproxy-bin`, which does not install a background proxy service.

The helper stops the proxy and removes its temporary configuration on success,
failure, or a handled termination signal. An independent process deadline stops
the proxy even if the helper is forcibly killed; a separate timeout also bounds
Vega's installation process. Each stage defaults to 15 minutes. Data Studio's
overall deployment timeout is separate and is not extended by this setting.
The proxy is not needed when starting or running the installed Robot Node.

Local proxy integration tests use loopback HTTP/HTTPS servers and do not contact
robots. With `tinyproxy`, `curl`, `openssl`, and pytest available, run
`python -m pytest tests/deploy/test_proxy.py` from the repository root.

`DOCKER_GITHUB_PAT` on the deployment server is passed transiently to the installer
for private SDK/Node downloads. GitHub SSH URLs are rewritten to HTTPS so they use
the proxy. Without a token, Vega needs existing GitHub HTTPS credentials; SSH keys
alone cannot authenticate this deployment path. An optional
`UV_DEFAULT_INDEX` is also forwarded when a private Python package index is used.
Neither is written into the startup script or Git configuration.

Deployment installs both Robotiq and SR drivers, serial permissions, and a dedicated
Python under `.python` with EtherCAT permissions. It does not change permissions
on a shared or system interpreter. Hardware selection happens at Configure.

Sync preserves `.venv`, `.python`, `third_party`, environment files, and `unit_config.json`.
Running `vega-loop-robot` sessions block deployment; stop Robot/UTI first.
Stop any old `vega-loop-relay` manually before starting Loop on Teleop's port
7448. The new service controls do not manage that relay.
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
| `preflight COMMIT` | Check the installed robot revision, SSH tools, and direct Loop endpoint |
| `start` | Start the Vega robot tmux with the manually configured Loop endpoint |
| `stop` | Stop the Vega robot tmux |
| `check-stopped` | Require the Vega robot tmux to be stopped |

The same commands can be run manually from the Teleop PC. Deployment saves paths
and session names in `.env.loop-service`; the robot SSH password remains in the
existing owner-only connection file. Start/stop implementation changes belong to
this repository; DS Layout only invokes the entrypoint.
A running process can wait for Loop to become available; registration alone does
not activate robot control.
