# RobotEnv Protocol Buffers

`robotenv.proto` defines the existing RobotEnv messages and gRPC service. The Loop
bridge reuses these messages for in-process arm control; it does not start a gRPC
server.

Generated modules are included in the installed package:

```python
from dexcontrol.core.robotenv_vega.proto import robotenv_pb2, robotenv_pb2_grpc
```

The Python files in this directory keep older `from proto import ...` imports
working from a checkout.

After editing `robotenv.proto`, regenerate the packaged modules with a compatible
`grpcio-tools` version installed in the active Python environment:

```bash
bash proto/compile_proto.sh
```
