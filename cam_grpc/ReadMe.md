## Setup Process
0. Copy contents of cam_grpc on both cam server and client (It is OK to test both on a same computer)

1. Install protobuf and grpcio
```bash
python2 -m pip install protobuf grpcio grpcio_tools
python3 -m pip install protobuf grpcio grpcio_tools
```

2. Compile protobuf
```bash
protoc -I=. --python_out=. RemoteCam.proto
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. RemoteCam.proto
```

3. Run ***grpc_server_test.ipynb*** on server PC.

4. Run ***grpc_client_test.ipynb*** on client PC.

5. Check client recieves image correctly.
