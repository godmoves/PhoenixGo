load("//:rules.bzl", "cc_proto_library")

cc_library(
    name = "tensorflow_dynamic",
    srcs = [
        "tensorflow/libtensorflow_framework.so",
        "tensorflow/libgrpc_runtime.so",
    ],
    hdrs = glob(
        ["tensorflow/**/*"],
        exclude = [
            "**/*.a",
            "**/*.so",
            "tensorflow/flatbuffers/**",
            "tensorflow/tensorflow/lite/**",
        ],
    ),
    includes = [
        "tensorflow",
        "tensorflow/external/nsync/public",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tf_zero_model",
    srcs = ["tf_zero_model.cc"],
    hdrs = ["tf_zero_model.h"],
    deps = [
        ":zero_model_base",
        ":checkpoint_utils",
        ":tensorflow_dynamic",
        "@com_github_google_glog//:glog",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "trt_zero_model",
    srcs = ["trt_zero_model.cc"],
    hdrs = ["trt_zero_model.h"],
    deps = [
        ":zero_model_base",
        "@boost//:filesystem",
        "@com_github_google_glog//:glog",
    ] + select({
        "//model/config:enable_trt": [
            "@local_config_cuda//:cuda_headers",
            "@local_config_cuda//:cudart",
            "@local_config_tensorrt//:nv_infer",
            "@local_config_tensorrt//:nvparsers",
        ],
        "//conditions:default": [],
    }),
    copts = select({
        "//model/config:enable_trt": ["-DUSE_TENSORRT=1"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tpu_zero_model",
    srcs = ["tpu_zero_model.cc"],
    hdrs = ["tpu_zero_model.h"],
    deps = [
        ":zero_model_base",
        ":tensorflow_dynamic",
        "@com_github_google_glog//:glog",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
	"@com_google_absl//absl/types:span",
	"@com_google_absl//absl/container:inlined_vector",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "zero_model_base",
    hdrs = ["zero_model_base.h"],
    deps = [
        ":model_config_cc_proto",
        "//common:errordef",
        "//common:go_comm",
    ],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "model_config_cc_proto",
    srcs = ["model_config.proto"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "checkpoint_utils",
    srcs = ["checkpoint_utils.cc"],
    hdrs = ["checkpoint_utils.h"],
    deps = [
        ":checkpoint_state_cc_proto",
        "@boost//:filesystem",
        "@com_github_google_glog//:glog",
    ],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "checkpoint_state_cc_proto",
    srcs = ["checkpoint_state.proto"],
)
