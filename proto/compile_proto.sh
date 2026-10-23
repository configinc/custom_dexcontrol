#!/bin/bash
# Compile RobotEnv and Vega BodyControl to Python code
#
# Required packages (must be compatible with protobuf==3.20.1):
#   pip install "grpcio>=1.44.0,<1.49.0" "grpcio-tools>=1.44.0,<1.49.0" "protobuf==3.20.1" "setuptools<81"
#
# NOTE: protobuf 3.20.1 is required for compatibility with polymetis (Franka).
#       grpcio-tools >= 1.50.0 requires protobuf >= 4.21.6 and is NOT compatible.

set -e

echo "Compiling RobotEnv and Vega BodyControl..."

python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./proto \
    --grpc_python_out=./proto \
    proto/robotenv.proto proto/vega_body.proto

echo "✓ Generated proto/robotenv_pb2.py"
echo "✓ Generated proto/robotenv_pb2_grpc.py"

# Fix imports in generated files
echo "Fixing imports..."
sed -i 's/^import robotenv_pb2/from proto import robotenv_pb2/' proto/robotenv_pb2_grpc.py
sed -i 's/^import vega_body_pb2/from proto import vega_body_pb2/' proto/vega_body_pb2_grpc.py
echo "✓ Fixed generated imports"

echo ""
echo "Protobuf compilation complete!"
