"""Run a command on the deployed Vega without exposing its SSH password in argv."""

import json
import os
from pathlib import Path
import shlex
import sys

connection = json.loads(Path(sys.argv[1]).read_text())
env = os.environ.copy()
env["SSHPASS"] = connection["password"]
command = [
    "sshpass", "-e", "ssh", "-T",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
    f"{connection['user']}@{connection['host']}",
    shlex.join(sys.argv[2:]),
]
os.execvpe(command[0], command, env)
