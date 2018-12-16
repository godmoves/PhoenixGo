#!/bin/bash

bazel build //mcts:mcts_main
bazel build //mcts:debug_tool
bazel build //dist:dist_zero_model_server