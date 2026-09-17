#!/usr/bin/env python3
"""Run a Vega installation over SSH with a temporary Teleop HTTP(S) proxy.

The installation script arrives on stdin, so credentials never enter argv or
temporary files. Only the Vega-facing interface and the Vega client are allowed.
"""

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def gateway_addresses(vega_host):
    vega_ip = socket.gethostbyname(vega_host)
    route = subprocess.run(
        ["ip", "-j", "-4", "route", "get", vega_ip],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    gateway_ip = json.loads(route.stdout)[0]["prefsrc"]
    return gateway_ip, vega_ip


def proxy_environment(url):
    values = {
        "http_proxy": url,
        "https_proxy": url,
        "HTTP_PROXY": url,
        "HTTPS_PROXY": url,
        "no_proxy": "localhost,127.0.0.1,::1",
        "NO_PROXY": "localhost,127.0.0.1,::1",
    }
    return "unset all_proxy ALL_PROXY\n" + "".join(
        f"export {key}={shlex.quote(value)}\n" for key, value in values.items()
    )


def stop_process(process):
    if process is None:
        return
    # Both children have their own session. Kill their descendants as well,
    # including tinyproxy under timeout and ssh under sshpass.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def wait_for_proxy(process, address):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("The temporary deployment proxy failed to start.")
        try:
            with socket.create_connection(address, timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("The temporary deployment proxy did not become ready.")


def run_install(vega_host, ssh_argv, script, timeout_seconds):
    gateway_ip, vega_ip = gateway_addresses(vega_host)
    with socket.socket() as reservation:
        reservation.bind((gateway_ip, 0))
        port = reservation.getsockname()[1]

    proxy = None
    install = None
    with tempfile.TemporaryDirectory(prefix="vega-deploy-proxy-") as directory:
        config = Path(directory) / "tinyproxy.conf"
        config.write_text(
            f"Listen {gateway_ip}\n"
            f"Port {port}\n"
            f"Allow {vega_ip}\n"
            "Timeout 60\n"
            "MaxClients 64\n"
            "LogLevel Critical\n"
            'LogFile "/dev/null"\n'
            f'PidFile "{directory}/tinyproxy.pid"\n'
        )
        deadline = time.monotonic() + timeout_seconds
        try:
            # An independent deadline also stops the proxy if this wrapper is
            # killed without running finally (e.g. the deployment is aborted).
            proxy = subprocess.Popen(
                [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=5s",
                    f"{timeout_seconds}s",
                    "tinyproxy",
                    "-d",
                    "-c",
                    str(config),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            wait_for_proxy(proxy, (gateway_ip, port))
            remaining = max(1, int(deadline - time.monotonic()))
            # Bound the remote process group too: closing SSH alone need not
            # stop a non-interactive apt/build process on Vega.
            install = subprocess.Popen(
                ssh_argv
                + [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=10s",
                    f"{remaining}s",
                    "bash",
                    "-s",
                ],
                stdin=subprocess.PIPE,
                start_new_session=True,
            )
            install.communicate(
                (proxy_environment(f"http://{gateway_ip}:{port}") + script).encode(),
                timeout=max(0.1, deadline - time.monotonic()),
            )
            return install.returncode
        except subprocess.TimeoutExpired:
            print(
                "Vega installation exceeded its deployment time limit.", file=sys.stderr
            )
            return 124
        finally:
            stop_process(install)
            stop_process(proxy)


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vega-host", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("ssh_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.ssh_argv
    if command[:1] == ["--"]:
        command = command[1:]
    if not command or args.timeout_seconds <= 0:
        parser.error("An SSH command and a positive timeout are required.")
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        return run_install(
            args.vega_host, command, sys.stdin.read(), args.timeout_seconds
        )
    except (
        OSError,
        ValueError,
        KeyError,
        subprocess.CalledProcessError,
        RuntimeError,
    ) as exc:
        print(f"Vega deployment proxy failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
