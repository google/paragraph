package(licenses = ["notice"])

cc_library(
    name = "simple_sim",
    srcs = [
        "simple_sim.cc",
    ],
    hdrs = [
        "simple_sim.h",
    ],
    deps = [
        "//paragraph/graph:graph",
        "//paragraph/shim:statusor",
        "//paragraph/scheduling:scheduler",
        "@com_google_absl//absl/memory:memory",
        "@com_google_absl//absl/strings:strings",
    ],
    visibility = [
        "//visibility:public",
    ],
)

cc_binary(
    name = "simulator",
    srcs = [
        "simulator.cc",
    ],
    deps = [
        ":simple_sim",
        "//paragraph/graph:graph",
        "//paragraph/shim:statusor",
        "//paragraph/scheduling:scheduler",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory:memory",
        "@com_google_absl//absl/strings:strings",
    ],
    visibility = [
        "//visibility:public",
    ],
)
cc_test(
    name = "simple_sim_test",
    size = "small",
    srcs = [
        "simple_sim_test.cc",
    ],
    deps = [
        ":simple_sim",
        "//paragraph/shim:test_macros",
        "//paragraph/translation:translation",
        "@com_google_absl//absl/status",
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
