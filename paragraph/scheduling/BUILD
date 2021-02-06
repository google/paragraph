package(licenses = ["notice"])

cc_library(
    name = "scheduler",
    srcs = [
        "graph_scheduler.cc",
        "instruction_fsm.cc",
        "logger.cc",
        "subroutine_fsm.cc",
    ],
    hdrs = [
        "graph_scheduler.h",
        "instruction_fsm.h",
        "logger.h",
        "subroutine_fsm.h",
    ],
    deps = [
        "//paragraph/graph:graph",
        "//paragraph/shim:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory:memory",
        "@com_google_absl//absl/strings:strings",
    ],
    visibility = [
        "//visibility:public",
    ],
)

cc_test(
    name = "scheduler_test",
    size = "small",
    srcs = [
        "graph_scheduler_test.cc",
        "instruction_fsm_test.cc",
        "logger_test.cc",
        "subroutine_fsm_test.cc",
    ],
    deps = [
        ":scheduler",
        "//paragraph/graph:graph",
        "//paragraph/shim:test_macros",
        "@com_google_absl//absl/status",
        "@com_google_protobuf//:protobuf",
        "@googletest//:gtest_main",
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
