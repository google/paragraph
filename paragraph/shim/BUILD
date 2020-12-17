cc_library(
    name = "statusor",
    hdrs = [
        "macros.h",
        "statusor.h",
        "statusor_internals.h",
    ],
    deps = [
        "@com_google_absl//absl/memory:memory",
        "@com_google_absl//absl/status:status",
    ],
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "test_macros",
    hdrs = [
        "test_macros.h",
    ],
    deps = [
        ":statusor",
        "@googletest//:gtest_main",
    ],
    visibility = [
        "//visibility:public",
    ],
)

cc_test(
    name = "statusor_test",
    size = "small",
    srcs = [
        "macros_test.cc",
        "statusor_test.cc",
    ],
    deps = [
        ":statusor",
        ":test_macros",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "test_macros_test",
    size = "small",
    srcs = [
        "test_macros_test.cc",
    ],
    deps = [
        ":test_macros",
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
