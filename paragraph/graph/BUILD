load("@rules_proto//proto:defs.bzl", "proto_library")

package(licenses = ["notice"])

proto_library(
    name = "graph_proto",
    srcs = ["graph.proto"],
    deps = [
    ],
)

cc_proto_library(
    name = "graph_cc_proto",
    deps = ["graph_proto"],
)

cc_library(
    name = "graph",
    srcs = [
        "graph.cc",
        "instruction.cc",
        "opcode.cc",
        "subroutine.cc",
    ],
    hdrs = [
        "graph.h",
        "instruction.h",
        "opcode.h",
        "subroutine.h",
    ],
    deps = [
        ":graph_cc_proto",
        "//paragraph/shim:statusor",
        "@com_google_protobuf//:protobuf",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory:memory",
        "@com_google_absl//absl/strings:strings",
    ],
    visibility = [
        "//visibility:public",
    ],
)

cc_binary(
    name = "graph_converter",
    srcs = ["graph_converter.cc"],
    deps = [
        ":graph",
        "//paragraph/shim:statusor",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_protobuf//:protobuf",
    ],
    visibility = [
        "//visibility:public",
    ],
)

cc_test(
    name = "graph_test",
    size = "small",
    srcs = [
        "graph_test.cc",
        "instruction_test.cc",
        "opcode_test.cc",
        "subroutine_test.cc",
    ],
    deps = [
        ":graph",
        "//paragraph/shim:test_macros",
        "//paragraph/translation:translation",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest_main",
        "@nlohmann_json//:nlohmann_json",
        "@com_google_protobuf//:protobuf",
    ],
)

genrule(
    name = "lint_cc",
    srcs = glob([
        "*.cc",
        "*.h",
        "*.tcc",
    ]),
    outs = ["linted"],
    cmd = """
        python $(location @cpplint//:cpplint) \
          --root=$$(pwd) \
          --headers=h,tcc \
          --extensions=cc,h,tcc \
          --quiet $(SRCS) > $@
        echo // $$(date) > $@
        """,
    tools = [
        "@cpplint",
    ],
)
