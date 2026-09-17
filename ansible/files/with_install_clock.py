#!/usr/bin/env python3
"""Use Teleop time only for an installation, then restore Vega time and NTP.

Executed on Vega with a JSON request on stdin. No NTP configuration, boot-time
service enablement, or hardware clock is changed. The outer deployment timeout
also terminates this wrapper if its SSH connection disappears.
"""

import json
import os
import signal
import subprocess
import sys
import time

TIME_SERVICES = (
    "chrony.service",
    "systemd-timesyncd.service",
    "ntp.service",
    "ntpsec.service",
)


def privileged(*args):
    subprocess.run(["sudo", "-n", *args], check=True, timeout=5)


def stop_install(process):
    if process is None:
        return
    # Stop descendants before restoring the clock, including background builds.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_install(epoch, script):
    active = [
        unit for unit in TIME_SERVICES
        if subprocess.run(
            ["systemctl", "is-active", "--quiet", unit], timeout=5
        ).returncode == 0
    ]
    original_epoch = time.time()
    started_at = time.monotonic()
    stopped = []
    clock_changed = False
    install = None
    result = 1
    try:
        for unit in active:
            # Include a partially completed/failed stop in restoration too.
            stopped.append(unit)
            privileged("systemctl", "stop", unit)
        clock_changed = True
        privileged("date", "--utc", "--set", f"@{epoch:.6f}")
        install = subprocess.Popen(
            ["bash", "-euo", "pipefail", "-s"],
            stdin=subprocess.PIPE,
            start_new_session=True,
        )
        install.communicate(script.encode())
        result = install.returncode
    finally:
        # A second termination signal must not interrupt restoration. SIGKILL
        # and power loss cannot be handled; no persistent settings were changed.
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        stop_install(install)
        if clock_changed:
            restored_epoch = original_epoch + time.monotonic() - started_at
            try:
                privileged("date", "--utc", "--set", f"@{restored_epoch:.6f}")
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"Failed to restore Vega clock: {exc}", file=sys.stderr)
                result = 1
        for unit in stopped:
            try:
                privileged("systemctl", "start", unit)
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"Failed to resume {unit}: {exc}", file=sys.stderr)
                result = 1
    return result


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    request = json.load(sys.stdin)
    epoch = float(request["epoch"])
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        return run_install(epoch, request["script"])
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Temporary installation clock failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
