cc_library(
    name = "sendrecv_interface",
    srcs = [
        "sendrecv_translator.cc",
    ],
    hdrs = [
        "sendrecv_translator.h",
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
    name = "sendrecv",
    srcs = [
        "push_sendrecv_translator.cc",
    ],
    hdrs = [
        "push_sendrecv_translator.h",
    ],
    deps = [
        "sendrecv_interface",
        "//paragraph/translation:factory",
    ],
    visibility = [
        "//paragraph/translation:__subpackages__",
    ],
    alwayslink = 1,
)

cc_test(
    name = "sendrecv_test",
    size = "small",
    srcs = [
        "push_sendrecv_translator_test.cc",
    ],
    deps = [
        ":sendrecv",
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
