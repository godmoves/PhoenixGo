load("@protobuf_archive//:protobuf.bzl", protobuf_cc_proto_library="cc_proto_library")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def cc_proto_library(name, srcs=[], deps=[], use_grpc_plugin=False, **kwargs):
    protobuf_cc_proto_library(
        name=name,
        srcs=srcs,
        deps=deps,
        use_grpc_plugin=use_grpc_plugin,
        cc_libs = ["@protobuf_archive//:protobuf"],
        protoc="@protobuf_archive//:protoc",
        default_runtime="@protobuf_archive//:protobuf",
        **kwargs
    )
