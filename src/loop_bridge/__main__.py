"""CLI for the external dual-arm Vega Robot Node."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from loop_sdk import NodeConnectionConfig

from loop_bridge.source_server import serve_dual_arm


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Expose both arms of one Vega as a Loop Node Graph Robot Node"
    )
    parser.add_argument("--node-id", required=True)
    parser.add_argument(
        "--loop-endpoint",
        default=os.environ.get("LOOP_NODE_GRAPH_NODE_ENDPOINT") or "tcp/127.0.0.1:7448",
        help="Loop address (default: LOOP_NODE_GRAPH_NODE_ENDPOINT or tcp/127.0.0.1:7448)",
    )
    parser.add_argument("--status-period-ms", type=_positive_int, default=250)
    parser.add_argument("--control-request-capacity", type=_positive_int, default=16)
    parser.add_argument("--data-request-capacity", type=_positive_int, default=16)

    args = parser.parse_args(argv)

    connection = NodeConnectionConfig(
        loop_endpoint=args.loop_endpoint,
        status_period_ms=args.status_period_ms,
        control_request_capacity=args.control_request_capacity,
        data_request_capacity=args.data_request_capacity,
    )
    serve_dual_arm(node_id=args.node_id, connection=connection)


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


if __name__ == "__main__":
    main()
