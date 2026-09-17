"""Exercise the service entrypoint with local SSH and robot-process fakes."""

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def service(tmp_path):
    teleop = tmp_path / "teleop"
    robot = tmp_path / "robot"
    commands = tmp_path / "bin"
    for path in (teleop / "ansible/files", robot / "ansible/files", commands):
        path.mkdir(parents=True)
    shutil.copy(ROOT / "loop-service.sh", teleop)
    shutil.copy(ROOT / "ansible/files/vega_ssh.py", teleop / "ansible/files")
    connection = teleop / "connection.json"
    connection.write_text(
        json.dumps({"host": "192.168.5.20", "user": "robot", "password": "fixture"})
    )
    settings = {
        "robot_project_dir": str(robot),
        "robot_start_script": str(robot / "start.sh"),
        "robot_session": "vega-loop-robot",
        "connection_file": str(connection),
    }
    (robot / "ansible/files/service_control.sh").write_text(
        """record() {
    python3 - "$@" <<'PY'
import json, os, sys
with open(os.environ['STUB_LOG'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
PY
}
require_stopped() { record check-stopped "$@"; return "${STUB_BUSY:-0}"; }
start_session() { record start "$@"; return "${STUB_START_RC:-0}"; }
stop_session() { record stop "$@"; return "${STUB_STOP_RC:-0}"; }
check_install() { record preflight "$@"; }
"""
    )
    fakes = {
        "sshpass": """#!/usr/bin/env python3
import shlex, subprocess, sys
# vega_ssh.py supplies one shell-quoted remote command. Execute it locally;
# stdin is the real remote service-control script, without any SSH connection.
sys.exit(subprocess.call(shlex.split(sys.argv[-1])))
""",
        "ip": "#!/bin/sh\necho 'Unexpected automatic route detection' >&2\nexit 99\n",
        "tmux": "#!/bin/sh\necho 'Unexpected tmux on Teleop' >&2\nexit 99\n",
        "socat": "#!/bin/sh\necho 'Unexpected relay on Teleop' >&2\nexit 99\n",
    }
    for name, content in fakes.items():
        script = commands / name
        script.write_text(content)
        script.chmod(0o755)
    log = tmp_path / "calls.jsonl"

    def run(*args, **overrides):
        (teleop / ".env.loop-service").write_text(
            "".join(f"{key}={shlex.quote(value)}\n" for key, value in settings.items())
        )
        log.write_text("")
        environment = dict(
            os.environ,
            PATH=f"{commands}:{os.environ['PATH']}",
            STUB_LOG=str(log),
            LOOP_NODE_GRAPH_NODE_ENDPOINT="tcp/192.168.5.17:7448",
        )
        environment.update(overrides)
        result = subprocess.run(
            ["bash", str(teleop / "loop-service.sh"), *args],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        return result, calls

    return run


@pytest.mark.parametrize("endpoint", ["", "tcp/0.0.0.0:7448", "tcp/previous-gpu:7448"])
def test_start_uses_vega_launcher_without_a_teleop_endpoint(service, endpoint):
    result, calls = service("start", LOOP_NODE_GRAPH_NODE_ENDPOINT=endpoint)
    assert result.returncode == 0, result.stderr
    assert [call[0] for call in calls] == ["check-stopped", "start"]
    assert len(calls[-1]) == 4  # start_session receives only session, project, script.


def test_busy_robot_is_not_restarted(service):
    result, calls = service("start", STUB_BUSY="1")
    assert result.returncode != 0
    assert [call[0] for call in calls] == ["check-stopped"]


def test_failed_start_cleans_up_only_the_robot(service):
    result, calls = service("start", STUB_START_RC="1")
    assert result.returncode != 0
    assert [call[0] for call in calls] == ["check-stopped", "start", "stop"]


def test_preflight_checks_robot_revision_without_relay(service):
    result, calls = service(
        "preflight", "fixture-commit", LOOP_NODE_GRAPH_NODE_ENDPOINT=""
    )
    assert result.returncode == 0, result.stderr
    assert [call[0] for call in calls] == ["preflight"]
    assert calls[0][2] == "fixture-commit"


@pytest.mark.parametrize("action", ["stop", "check-stopped"])
def test_stop_controls_do_not_require_a_loop_endpoint(service, action):
    result, calls = service(action, LOOP_NODE_GRAPH_NODE_ENDPOINT="")
    assert result.returncode == 0, result.stderr
    assert [call[0] for call in calls] == [action]


def test_stop_failure_is_reported(service):
    result, calls = service("stop", STUB_STOP_RC="1")
    assert result.returncode != 0
    assert [call[0] for call in calls] == ["stop"]
