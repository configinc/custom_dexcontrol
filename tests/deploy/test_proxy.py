"""Local integration tests; no robot, SSH server, or external network is used.

Requires tinyproxy, curl, and openssl on PATH. The SSH transport is replaced by
env, so the same bounded installation command executes against local fixtures.
"""

import http.server
import json
import os
import select
import shlex
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import pytest

HELPER = Path(__file__).resolve().parents[2] / "ansible/files/with_deploy_proxy.py"
pytestmark = pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("tinyproxy", "curl", "openssl")),
    reason="Local proxy integration tests require tinyproxy, curl, and openssl",
)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"downloaded through Teleop"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def origins(tmp_path):
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    http_origin = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    https_origin = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    https_origin.socket = context.wrap_socket(https_origin.socket, server_side=True)
    threads = []
    for server in (http_origin, https_origin):
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        threads.append(thread)
    try:
        yield http_origin.server_port, https_origin.server_port, cert
    finally:
        for server, thread in zip((http_origin, https_origin), threads):
            server.shutdown()
            server.server_close()
            thread.join()


@pytest.fixture
def invocation(tmp_path):
    def make(timeout=10):
        command = [
            sys.executable,
            str(HELPER),
            "--vega-host",
            "127.0.0.1",
            "--timeout-seconds",
            str(timeout),
            "--",
            "env",
        ]
        # Existing proxy settings must not bypass the deployment proxy.
        environment = dict(
            os.environ,
            TMPDIR=str(tmp_path),
            http_proxy="http://invalid:1",
            HTTPS_PROXY="http://invalid:1",
            no_proxy="*",
            ALL_PROXY="http://invalid:1",
        )
        return command, environment

    return make


def assert_closed(url, timeout=2):
    address = urlsplit(url)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(
                (address.hostname, address.port), timeout=0.1
            ):
                pass
        except OSError:
            return
        time.sleep(0.05)
    pytest.fail("The deployment proxy is still listening")


def assert_clean(tmp_path):
    assert not list(tmp_path.glob("vega-deploy-proxy-*"))


def test_http_https_downloads_and_environment(invocation, origins, tmp_path):
    http_port, https_port, cert = origins
    command, environment = invocation()
    script = f"""set -euo pipefail
python3 -c 'import os, json; print(json.dumps({{k:v for k,v in os.environ.items() if "proxy" in k.lower()}}))'
curl --noproxy '' --fail --silent --show-error --max-time 3 http://localhost:{http_port}/package
curl --noproxy '' --fail --silent --show-error --max-time 3 --cacert {shlex.quote(str(cert))} https://localhost:{https_port}/package
"""
    result = subprocess.run(
        command,
        input=script,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    settings, body = result.stdout.split("\n", 1)
    settings = json.loads(settings)
    url = settings["http_proxy"]
    assert all(
        settings[key] == url for key in ("HTTP_PROXY", "HTTPS_PROXY", "https_proxy")
    )
    assert settings["no_proxy"] == settings["NO_PROXY"] == "localhost,127.0.0.1,::1"
    assert "ALL_PROXY" not in settings and "all_proxy" not in settings
    assert body == "downloaded through Teleop" * 2
    assert_closed(url)
    assert_clean(tmp_path)


@pytest.mark.parametrize(
    ("script", "timeout", "expected"),
    [
        ('echo "$http_proxy"; exit 37', 10, 37),
        ('echo "$http_proxy"; exec sleep 30', 2, 124),
    ],
)
def test_failure_and_timeout_cleanup(invocation, tmp_path, script, timeout, expected):
    command, environment = invocation(timeout)
    result = subprocess.run(
        command,
        input=script,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == expected, result.stderr
    assert_closed(result.stdout.strip())
    assert_clean(tmp_path)


@pytest.mark.parametrize(
    "sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGKILL]
)
def test_interrupted_deploy_closes_proxy(invocation, tmp_path, sig):
    command, environment = invocation(3 if sig == signal.SIGKILL else 10)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        text=True,
    )
    try:
        process.stdin.write('echo "$http_proxy"; exec sleep 30\n')
        process.stdin.close()
        assert select.select([process.stdout], [], [], 5)[0], (
            "Installation did not start"
        )
        url = process.stdout.readline().strip()
        assert url.startswith("http://"), process.stderr.read()
        process.send_signal(sig)
        assert process.wait(timeout=10) == (
            -sig if sig == signal.SIGKILL else 128 + sig
        )
        assert_closed(url, timeout=8)
        if sig != signal.SIGKILL:
            assert_clean(tmp_path)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()


def test_proxy_rejects_clients_other_than_vega(invocation, origins):
    http_port, _, _ = origins
    command, environment = invocation()
    script = f"""python3 - <<'PY'
import os, socket
from urllib.parse import urlsplit
url = urlsplit(os.environ['http_proxy'])
with socket.create_connection((url.hostname, url.port), source_address=('127.0.0.2', 0)) as connection:
    connection.sendall(b'GET http://localhost:{http_port}/ HTTP/1.0\\r\\n\\r\\n')
    print(connection.recv(4096).split(b'\\r\\n')[0].decode())
PY
"""
    result = subprocess.run(
        command,
        input=script,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "403" in result.stdout


def test_proxy_startup_failure_does_not_run_install(invocation, tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    proxy = fake_bin / "tinyproxy"
    proxy.write_text("#!/bin/sh\nexit 1\n")
    proxy.chmod(0o755)
    command, environment = invocation()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    marker = tmp_path / "install-started"
    result = subprocess.run(
        command,
        input=f"touch {shlex.quote(str(marker))}",
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert not marker.exists()
    assert_clean(tmp_path)
