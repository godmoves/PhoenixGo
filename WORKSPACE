load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "a38539c5b5c358548e75b44141b4ab637bba7c4dc02b46b1f62a96d6433f56ae",
    strip_prefix = "rules_closure-dbb96841cc0a5fb2664c37822803b06dab20c7d1",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",  # 2018-04-13
    ],
)

http_archive(
    name = "com_google_absl",
    strip_prefix = "abseil-cpp-5441bbe1db5d0f2ca24b5b60166367b0966790af",
    urls = ["https://github.com/abseil/abseil-cpp/archive/5441bbe1db5d0f2ca24b5b60166367b0966790af.zip"],
)

http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "6e16c8bc91b1310a44f3965e616383dbda48f83e8c1eaa2370a215057b00cabe",
    strip_prefix = "gflags-77592648e3f3be87d6c7123eb81cbad75f9aef5a",
    urls = [
        "https://mirror.bazel.build/github.com/gflags/gflags/archive/77592648e3f3be87d6c7123eb81cbad75f9aef5a.tar.gz",
        "https://github.com/gflags/gflags/archive/77592648e3f3be87d6c7123eb81cbad75f9aef5a.tar.gz",
    ],
)

http_archive(
    name = "com_github_google_glog",
    urls = ["https://github.com/google/glog/archive/55cc27b6eca3d7906fc1a920ca95df7717deb4e7.tar.gz"],
    sha256 = "4966f4233e4dcac53c7c7dd26054d17c045447e183bf9df7081575d2b888b95b",
    strip_prefix = "glog-55cc27b6eca3d7906fc1a920ca95df7717deb4e7",
    patches = ["//third_party/glog:glog.patch"],
)

http_archive(
    name = "com_github_nelhage_rules_boost",
    urls = ["https://github.com/nelhage/rules_boost/archive/c1975a9a45c97823ae9e68fbccb821418099168f.tar.gz"],
    sha256 = "ab8403986bea12da70c1ed86ef688e72c3062dd6b6a9a84035c4f13145defb93",
    strip_prefix = "rules_boost-c1975a9a45c97823ae9e68fbccb821418099168f",
)

load("//:workspace.bzl", "tf_workspace")
tf_workspace()

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

load("//model/config:cuda_configure.bzl", "cuda_configure")
load("//model/config:tensorrt_configure.bzl", "tensorrt_configure")

cuda_configure(name = "local_config_cuda")

tensorrt_configure(name = "local_config_tensorrt")
