#!/bin/bash

cd "`dirname $0`/.."

bazel build //mcts:mcts_main
bazel build //mcts:debug_tool
bazel build //dist:dist_zero_model_server
bazel build //model:build_tensorrt_model