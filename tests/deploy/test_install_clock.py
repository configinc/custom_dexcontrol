"""Exercise temporary clock restoration without changing any system clocks."""

import json
import os
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

FILES = Path(__file__).resolve().parents[2] / "ansible/files"
HELPER = FILES / "with_install_clock.py"
GATEWAY_EPOCH = 2_000_000_000.0


@pytest.fixture
def clock_commands(tmp_path):
    commands = tmp_path / "bin"
    commands.mkdir()
    log = tmp_path / "clock.jsonl"
    log.write_text("")
    fake = commands / "clock-command"
    fake.write_text(
        """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if Path(sys.argv[0]).name == 'systemctl':
    assert args[:2] == ['is-active', '--quiet'], args
    active = args[2] == 'chrony.service' and os.environ.get('CLOCK_INACTIVE') != '1'
    sys.exit(0 if active else 3)
assert args.pop(0) == '-n', args
log = Path(os.environ['CLOCK_LOG'])
previous = [json.loads(line) for line in log.read_text().splitlines()]
with log.open('a') as f:
    f.write(json.dumps(args) + '\\n')
if args[0] == 'systemctl':
    assert args[1] in ('stop', 'start'), args
    operation = args[1]
else:
    assert args[:3] == ['date', '--utc', '--set'], args
    operation = 'restore' if any(a[0] == 'date' for a in previous) else 'set'
sys.exit(42 if os.environ.get('CLOCK_FAIL') == operation else 0)
"""
    )
    fake.chmod(0o755)
    for name in ("sudo", "systemctl"):
        (commands / name).symlink_to(fake)
    environment = dict(
        os.environ,
        PATH=f"{commands}:{os.environ['PATH']}",
        CLOCK_LOG=str(log),
    )
    return environment, log


def calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def request(script):
    return json.dumps({"epoch": GATEWAY_EPOCH, "script": script})


def assert_restored(log, before, active=True):
    recorded = calls(log)
    dates = [call for call in recorded if call[0] == "date"]
    assert len(dates) == 2
    assert float(dates[0][-1][1:]) == GATEWAY_EPOCH
    # Restoration follows elapsed monotonic time, not the temporary clock.
    assert before <= float(dates[1][-1][1:]) <= time.time()
    if active:
        assert recorded[0] == ["systemctl", "stop", "chrony.service"]
        assert recorded[-1] == ["systemctl", "start", "chrony.service"]
    else:
        assert all(call[0] == "date" for call in recorded)


@pytest.mark.parametrize("active", [True, False])
@pytest.mark.parametrize("exit_code", [0, 37])
def test_install_restores_time_and_original_service_state(
    clock_commands, active, exit_code
):
    environment, log = clock_commands
    environment["CLOCK_INACTIVE"] = "0" if active else "1"
    before = time.time()
    result = subprocess.run(
        [sys.executable, str(HELPER)],
        input=request(f"echo installed; sleep 0.1; exit {exit_code}"),
        env=environment, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == exit_code, result.stderr
    assert result.stdout.strip() == "installed"
    assert_restored(log, before, active)


@pytest.mark.parametrize("failure", ["stop", "set", "restore", "start"])
def test_setup_and_restore_failures_resume_ntp_and_fail_deployment(
    clock_commands, failure
):
    environment, log = clock_commands
    environment["CLOCK_FAIL"] = failure
    result = subprocess.run(
        [sys.executable, str(HELPER)], input=request("echo installed"),
        env=environment, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert ("installed" in result.stdout) == (failure in ("restore", "start"))
    assert calls(log)[-1] == ["systemctl", "start", "chrony.service"]


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_interruption_restores_clock_and_stops_installer(clock_commands, sig):
    environment, log = clock_commands
    before = time.time()
    process = subprocess.Popen(
        [sys.executable, str(HELPER)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=environment, text=True,
    )
    try:
        process.stdin.write(request("echo $$; exec sleep 30"))
        process.stdin.close()
        assert select.select([process.stdout], [], [], 5)[0]
        installer_pid = int(process.stdout.readline())
        process.send_signal(sig)
        assert process.wait(timeout=10) == 128 + sig
        assert_restored(log, before)
        with pytest.raises(ProcessLookupError):
            os.kill(installer_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()


def test_remote_timeout_restores_clock(clock_commands):
    environment, log = clock_commands
    before = time.time()
    result = subprocess.run(
        ["timeout", "--kill-after=5s", "1s", sys.executable, str(HELPER)],
        input=request("echo installed; exec sleep 30"),
        env=environment, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 124, result.stderr
    assert result.stdout.strip() == "installed"
    assert_restored(log, before)


@pytest.mark.skipif(shutil.which("tinyproxy") is None, reason="requires tinyproxy")
def test_proxy_wraps_installation_with_temporary_clock(clock_commands):
    environment, log = clock_commands
    before = time.time()
    result = subprocess.run(
        [sys.executable, str(FILES / "with_deploy_proxy.py"),
         "--vega-host", "127.0.0.1", "--temporary-clock",
         "--timeout-seconds", "10", "--", "env"],
        input='printf "installed\\n"; test -n "$http_proxy"; exit 37',
        env=environment, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 37, result.stderr
    assert result.stdout.strip() == "installed"
    recorded = calls(log)
    dates = [call for call in recorded if call[0] == "date"]
    assert len(dates) == 2
    assert all(before <= float(call[-1][1:]) <= time.time() for call in dates)
    assert recorded[0] == ["systemctl", "stop", "chrony.service"]
    assert recorded[-1] == ["systemctl", "start", "chrony.service"]


@pytest.mark.skipif(shutil.which("tinyproxy") is None, reason="requires tinyproxy")
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_gateway_interruption_still_restores_remote_clock(clock_commands, sig):
    environment, log = clock_commands
    process = subprocess.Popen(
        [sys.executable, str(FILES / "with_deploy_proxy.py"),
         "--vega-host", "127.0.0.1", "--temporary-clock",
         "--timeout-seconds", "3", "--", "env"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=environment, text=True,
    )
    try:
        process.stdin.write("echo ready; exec sleep 30\n")
        process.stdin.close()
        assert select.select([process.stdout], [], [], 5)[0]
        assert process.stdout.readline().strip() == "ready"
        process.send_signal(sig)
        assert process.wait(timeout=10) == (
            -sig if sig == signal.SIGKILL else 128 + sig
        )
        # The remote timeout survives even a killed gateway wrapper.
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            recorded = calls(log)
            if recorded[-1] == ["systemctl", "start", "chrony.service"]:
                break
            time.sleep(0.05)
        else:
            pytest.fail("Remote NTP service was not restored")
        assert len([call for call in recorded if call[0] == "date"]) == 2
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()
