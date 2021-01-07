package(licenses = ["notice"])

cc_library(
    name = "allgather_interface",
    srcs = [
        "allgather_translator.cc",
    ],
    hdrs = [
        "allgather_translator.h",
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
    name = "allgather",
    srcs = [
        "unidir_ring_allgather_translator.cc",
    ],
    hdrs = [
        "unidir_ring_allgather_translator.h",
    ],
    deps = [
        ":allgather_interface",
        "//paragraph/translation:factory",
        "//paragraph/translation/barrier:barrier",
    ],
    visibility = [
        "//paragraph/translation:__subpackages__",
    ],
    alwayslink = 1,
)

cc_test(
    name = "allgather_test",
    size = "small",
    srcs = [
        "unidir_ring_allgather_translator_test.cc",
    ],
    deps = [
        ":allgather",
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
