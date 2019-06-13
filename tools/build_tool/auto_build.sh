#!/bin/bash

# cd to parent of script's directory (i.e. project root)
cd "${0%/*}"/../..

if [ $1 == "trt" ]; then
  bazel build --define=grpc_no_ares=true --define=trt=1 //mcts:mcts_main
  bazel build --define=trt=1 //mcts:debug_tool
  bazel build --define=grpc_no_ares=true --define=trt=1 //dist:dist_zero_model_server
else
  bazel build --define=grpc_no_ares=true //mcts:mcts_main
  bazel build //mcts:debug_tool
  bazel build --define=grpc_no_ares=true //dist:dist_zero_model_server
fi
