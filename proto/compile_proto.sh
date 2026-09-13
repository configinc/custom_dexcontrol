#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output="$project_dir/src/dexcontrol/core/robotenv_vega/proto"
# Install grpcio-tools compatible with the runtime before regenerating.
python -m grpc_tools.protoc -I"$project_dir/proto" \
    --python_out="$output" --grpc_python_out="$output" \
    "$project_dir/proto/robotenv.proto"
sed -i 's/^import robotenv_pb2/from dexcontrol.core.robotenv_vega.proto import robotenv_pb2/' "$output/robotenv_pb2_grpc.py"
