load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

hash = "582c8d2"
http_archive(
  name = "org_tensorflow",
  urls = [
      "https://github.com/tensorflow/tensorflow/tarball/" + hash,
  ],
  type = "tar.gz",
  strip_prefix = "tensorflow-tensorflow-" + hash,
  patch_args = ["-p1"],
  patches = [
      "//paragraph/bridging/hlo/tensorflow_patches:build.patch",
      "//paragraph/bridging/hlo/tensorflow_patches:tf.patch",
  ],
)

# rules_cc defines rules for generating C++ code from Protocol Buffers.

# rules_proto defines abstract rules for building Protocol Buffers.
hash = "97d8af4"
http_archive(
    name = "rules_proto",
    urls = [
        "https://github.com/bazelbuild/rules_proto/tarball/" + hash,
    ],
    type = "tar.gz",
    strip_prefix = "bazelbuild-rules_proto-" + hash,
)
load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
rules_proto_dependencies()
rules_proto_toolchains()

release = "1.10.0"
http_archive(
  name = "googletest",
  urls = ["https://github.com/google/googletest/archive/release-" + release + ".tar.gz"],
  strip_prefix = "googletest-release-" + release,
)

http_file(
  name = "cpplint_build",
  urls = ["https://raw.githubusercontent.com/nicmcd/pkgbuild/master/cpplint.BUILD"],
)

release = "1.5.2"
http_archive(
    name = "cpplint",
    urls = ["https://github.com/cpplint/cpplint/archive/" + release + ".tar.gz"],
    strip_prefix = "cpplint-" + release,
    build_file = "@cpplint_build//file:downloaded",
)

hash = "c51510d"
http_archive(
    name = "com_google_absl",
    urls = ["https://github.com/abseil/abseil-cpp/tarball/" + hash],
    type = "tar.gz",
    strip_prefix = "abseil-abseil-cpp-" + hash,
)

hash = "46865ff"
http_archive(
  name = "libfactory",
  urls = ["https://github.com/nicmcd/libfactory/tarball/" + hash],
  type = "tar.gz",
  strip_prefix = "nicmcd-libfactory-" + hash,
)

http_file(
    name = "nlohmann_json_build",
    urls = ["https://raw.githubusercontent.com/nicmcd/pkgbuild/master/nlohmannjson.BUILD"],
)

release = "3.9.1"
http_archive(
    name = "nlohmann_json",
    urls = ["https://github.com/nlohmann/json/archive/v" + release + ".tar.gz"],
    strip_prefix = "json-" + release,
    build_file = "@nlohmann_json_build//file:downloaded",
)

# Tensorflow rules
# https://github.com/bazelbuild/rules_closure/tarball/308b05b
hash = "308b05b"
http_archive(
  name = "io_bazel_rules_closure",
  urls = ["https://github.com/bazelbuild/rules_closure/tarball/" + hash],
  type = "tar.gz",
  strip_prefix = "bazelbuild-rules_closure-" + hash,
)

# Load tf_repositories() before loading dependencies for other repository so
# that dependencies like com_google_protobuf won't be overridden.
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_repositories")
# Please add all new TensorFlow dependencies in workspace.bzl.
tf_repositories()

register_toolchains("@local_config_python//:py_toolchain")

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")

closure_repositories()

load("@org_tensorflow//third_party/toolchains/preconfig/generate:archives.bzl",
     "bazel_toolchains_archive")

bazel_toolchains_archive()

load(
    "@bazel_toolchains//repositories:repositories.bzl",
    bazel_toolchains_repositories = "repositories",
)

bazel_toolchains_repositories()

# Use `swift_rules_dependencies` to fetch the toolchains. With the
# `git_repository` rules above, the following call will skip redefining them.
load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")
swift_rules_dependencies()

# We must check the bazel version before trying to parse any other BUILD
# files, in case the parsing of those build files depends on the bazel
# version we require here.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")
check_bazel_version_at_least("1.0.0")

# If a target is bound twice, the later one wins, so we have to do tf bindings
# at the end of the WORKSPACE file.
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_bind")
tf_bind()

# Required for dependency @com_github_grpc_grpc

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

load("@org_tensorflow//third_party/googleapis:repository_rules.bzl", "config_googleapis")

config_googleapis()
