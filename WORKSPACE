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
    name = "com_github_google_glog",
    urls = ["https://github.com/google/glog/archive/55cc27b6eca3d7906fc1a920ca95df7717deb4e7.tar.gz"],
    sha256 = "4966f4233e4dcac53c7c7dd26054d17c045447e183bf9df7081575d2b888b95b",
    strip_prefix = "glog-55cc27b6eca3d7906fc1a920ca95df7717deb4e7",
    patches = ["//third_party/glog:glog.patch"],
)

http_archive(
    name = "org_tensorflow",
    urls = ["https://github.com/tensorflow/tensorflow/archive/v1.12.0.tar.gz"],
    sha256 = "3c87b81e37d4ed7f3da6200474fa5e656ffd20d8811068572f43610cae97ca92",
    strip_prefix = "tensorflow-1.12.0",
    patches = ["//third_party/tensorflow:tensorflow.patch"],
)

load('@org_tensorflow//tensorflow:workspace.bzl', 'tf_workspace')
tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")

http_archive(
    name = "com_github_nelhage_rules_boost",
    urls = ["https://github.com/nelhage/rules_boost/archive/c1975a9a45c97823ae9e68fbccb821418099168f.tar.gz"],
    sha256 = "ab8403986bea12da70c1ed86ef688e72c3062dd6b6a9a84035c4f13145defb93",
    strip_prefix = "rules_boost-c1975a9a45c97823ae9e68fbccb821418099168f",
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()
