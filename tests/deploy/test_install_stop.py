"""Exercise bootstrap shutdown with fake tmux sessions and no installed runtime."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[2] / "ansible/files/service_control.sh"
SESSIONS = ("vega-loop-robot", "robot-server")


@pytest.fixture
def shutdown(tmp_path):
    commands = tmp_path / "bin"
    commands.mkdir()
    state = tmp_path / "sessions.json"
    log = tmp_path / "calls.jsonl"
    tmux = commands / "tmux"
    tmux.write_text(
        f"#!{sys.executable}\n"
        """import json, os, sys
from pathlib import Path
args = sys.argv[1:]
state = Path(os.environ['STOP_TEST_STATE'])
sessions = json.loads(state.read_text())
with open(os.environ['STOP_TEST_LOG'], 'a') as stream:
    stream.write(json.dumps(args) + '\\n')
session = args[args.index('-t') + 1].lstrip('=%')
operation = args[0]
if operation == 'has-session':
    sys.exit(0 if session in sessions else 1)
assert session in sessions, args
if operation == 'list-panes':
    print('%' + session if args[-1] == '#{pane_id}' else sessions[session]['command'])
elif operation == 'send-keys':
    assert args[-1] == 'C-c', args
    behavior = sessions[session]['behavior']
    if behavior == 'exit':
        del sessions[session]
    elif behavior == 'shell':
        sessions[session]['command'] = 'bash'
elif operation == 'kill-session':
    assert sessions[session]['command'] == 'bash', 'Killed a live controller'
    del sessions[session]
else:
    raise AssertionError(args)
state.write_text(json.dumps(sessions))
"""
    )
    tmux.chmod(0o755)
    sleep = commands / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)

    def run(behaviors):
        state.write_text(
            json.dumps(
                {
                    name: {"command": "python3", "behavior": behavior}
                    for name, behavior in behaviors.items()
                }
            )
        )
        log.write_text("")
        script = HELPER.read_text() + "\nstop_sessions " + shlex.join(SESSIONS)
        result = subprocess.run(
            ["bash", "-s"],
            input=script,
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{commands}:{os.environ['PATH']}",
                "STOP_TEST_STATE": str(state),
                "STOP_TEST_LOG": str(log),
            },
            capture_output=True,
            text=True,
            timeout=15,
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        return result, json.loads(state.read_text()), calls

    return run


def test_first_install_stops_legacy_without_an_installed_loop_repo(shutdown):
    result, remaining, calls = shutdown({"robot-server": "shell"})

    assert result.returncode == 0, result.stderr
    assert not remaining
    assert result.stdout.splitlines() == ["Stopped robot-server"]
    assert ["send-keys", "-t", "%robot-server", "C-c"] in calls
    assert ["kill-session", "-t", "=robot-server"] in calls


def test_stop_handles_both_robot_sessions_without_touching_other_services(shutdown):
    result, remaining, _ = shutdown(
        {"vega-loop-robot": "exit", "robot-server": "shell", "unrelated": "stuck"}
    )

    assert result.returncode == 0, result.stderr
    assert set(remaining) == {"unrelated"}
    assert result.stdout.splitlines() == [f"Stopped {session}" for session in SESSIONS]


def test_stop_is_unchanged_when_no_robot_sessions_are_running(shutdown):
    result, remaining, calls = shutdown({})

    assert result.returncode == 0, result.stderr
    assert not remaining
    assert result.stdout == ""
    assert all(call[0] == "has-session" for call in calls)


@pytest.mark.parametrize("stuck", SESSIONS)
def test_shutdown_timeout_fails_but_still_stops_the_other_robot_session(
    shutdown, stuck
):
    result, remaining, calls = shutdown(
        {session: "stuck" if session == stuck else "exit" for session in SESSIONS}
    )

    assert result.returncode != 0
    assert set(remaining) == {stuck}
    assert f"{stuck} did not finish shutdown" in result.stderr
    assert {call[2] for call in calls if call[0] == "send-keys"} == {
        f"%{session}" for session in SESSIONS
    }
    assert not any(call[0] == "kill-session" for call in calls)
