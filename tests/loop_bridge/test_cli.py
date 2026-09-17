"""CLI connection defaults without starting a robot or importing its drivers."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    ("environment", "arguments", "expected"),
    [
        ("", [], "tcp/192.168.5.17:7448"),
        ("tcp/custom-teleop:7448", [], "tcp/custom-teleop:7448"),
        (
            "tcp/previous-host:7448",
            ["--loop-endpoint", "tcp/teleop:17448"],
            "tcp/teleop:17448",
        ),
    ],
)
def test_loop_endpoint_selection(monkeypatch, environment, arguments, expected):
    serve = Mock()
    monkeypatch.setenv("LOOP_NODE_GRAPH_NODE_ENDPOINT", environment)
    monkeypatch.setitem(
        sys.modules, "loop_sdk", SimpleNamespace(NodeConnectionConfig=SimpleNamespace)
    )
    monkeypatch.setitem(
        sys.modules, "loop_bridge.source_server", SimpleNamespace(serve_dual_arm=serve)
    )
    source = Path(__file__).resolve().parents[2] / "src/loop_bridge/__main__.py"
    spec = importlib.util.spec_from_file_location("vega_cli", source)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    cli.main(["--node-id", "robot", *arguments])

    assert serve.call_args.kwargs["node_id"] == "robot"
    assert serve.call_args.kwargs["connection"].loop_endpoint == expected
