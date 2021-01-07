package(licenses = ["notice"])

cc_library(
    name = "barrier_interface",
    srcs = [
        "barrier_translator.cc",
    ],
    hdrs = [
        "barrier_translator.h",
    ],
    deps = [
        "//paragraph/translation:translator",
        "@com_google_absl//absl/memory:memory",
        "@libfactory//:factory",
        "@nlohmann_json//:nlohmann_json",
    ],
    visibility = [
        "//paragraph/translation:__pkg__",
    ],
    alwayslink = 1,
)

cc_library(
    name = "barrier",
    srcs = [
        "centralized_barrier_translator.cc",
    ],
    hdrs = [
        "centralized_barrier_translator.h",
    ],
    deps = [
        ":barrier_interface",
        "//paragraph/translation:factory",
    ],
    visibility = [
        "//paragraph/translation:__subpackages__",
    ],
    alwayslink = 1,
)

cc_test(
    name = "barrier_test",
    size = "small",
    srcs = [
        "centralized_barrier_translator_test.cc",
    ],
    deps = [
        ":barrier",
        "//paragraph/shim:test_macros",
        "@googletest//:gtest_main",
    ],
    visibility = [
        "//paragraph/translation:__subpackages__",
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
