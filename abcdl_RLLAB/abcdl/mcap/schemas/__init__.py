"""ABC custom protobuf messages (RobotState/GripperState/Instructions/Annotation).

`abc_pb2.py` is generated from abc.proto and committed. Regenerate after editing
abc.proto:
    python -m grpc_tools.protoc -I abcdl/mcap/schemas \
        -I "$(python -c 'import grpc_tools,os;print(os.path.dirname(grpc_tools.__file__)+"/_proto")')" \
        --python_out abcdl/mcap/schemas abcdl/mcap/schemas/abc.proto
"""

from __future__ import annotations

from abcdl.mcap.schemas import abc_pb2

RobotState = abc_pb2.RobotState
GripperState = abc_pb2.GripperState
Instructions = abc_pb2.Instructions
Annotation = abc_pb2.Annotation
