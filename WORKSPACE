load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

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
