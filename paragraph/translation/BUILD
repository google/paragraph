package(licenses = ["notice"])

cc_library(
    name = "factory",
    srcs = [
        "translation_map.cc",
    ],
    hdrs = [
        "translation_map.h",
    ],
    deps = [
        ":translator",
        "//paragraph/graph:graph",
        "//paragraph/shim:statusor",
        "//paragraph/translation/allgather:allgather_interface",
        "//paragraph/translation/allreduce:allreduce_interface",
        "//paragraph/translation/barrier:barrier_interface",
        "//paragraph/translation/recv:recv_interface",
        "//paragraph/translation/reducescatter:reducescatter_interface",
        "//paragraph/translation/send:send_interface",
        "//paragraph/translation/sendrecv:sendrecv_interface",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory:memory",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:nlohmann_json",
    ],
    visibility = [
        "//paragraph/translation:__subpackages__",
    ],
    alwayslink = 1,
)

cc_library(
    name = "translator",
    srcs = [
    ],
    hdrs = [
        "translator.h",
    ],
    deps = [
        "//paragraph/graph:graph",
        "//paragraph/shim:statusor",
    ],
    visibility = [
        "//paragraph/translation:__subpackages__",
    ],
    alwayslink = 1,
)

cc_library(
    name = "translation",
    srcs = [
        "translation.cc",
    ],
    hdrs = [
        "translation.h",
    ],
    deps = [
        ":factory",
        "//paragraph/translation/allgather:allgather",
        "//paragraph/translation/allreduce:allreduce",
        "//paragraph/translation/barrier:barrier",
        "//paragraph/translation/recv:recv",
        "//paragraph/translation/reducescatter:reducescatter",
        "//paragraph/translation/send:send",
        "//paragraph/translation/sendrecv:sendrecv",
        "//paragraph/graph:graph",
        "//paragraph/shim:statusor",
        "@com_google_absl//absl/memory:memory",
        "@nlohmann_json//:nlohmann_json",
    ],
    visibility = [
        "//visibility:public",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "graph_translator",
    srcs = [
        "graph_translator.cc",
    ],
    deps = [
        ":translation",
        "//paragraph/graph:graph",
        "//paragraph/shim:statusor",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
        "@nlohmann_json//:nlohmann_json",
    ],
    visibility = [
        "//visibility:public",
    ],
)

cc_test(
    name = "translation_test",
    size = "small",
    srcs = [
        "translation_test.cc"
    ],
    deps = [
        ":translation",
        "//paragraph/shim:test_macros",
        "//paragraph/translation:factory",
        "@com_google_protobuf//:protobuf",
        "@googletest//:gtest_main",
        "@nlohmann_json//:nlohmann_json",
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
