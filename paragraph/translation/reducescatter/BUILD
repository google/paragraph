package(licenses = ["notice"])

cc_library(
    name = "reducescatter_interface",
    srcs = [
        "reducescatter_translator.cc",
    ],
    hdrs = [
        "reducescatter_translator.h",
    ],
    deps = [
        "//paragraph/translation:translator",
        "@com_google_absl//absl/memory:memory",
        "@libfactory//:factory",
        "@nlohmann_json//:nlohmann_json",
    ],
    visibility = [
        "//paragraph/translation:__subpackages__",
    ],
    alwayslink = 1,
)

cc_library(
    name = "reducescatter",
    srcs = [
        "bidir_ring_reducescatter_translator.cc",
        "mesh_1d_reducescatter_translator.cc",
        "mesh_2d_reducescatter_translator.cc",
        "ring_over_2d_grid_reducescatter_translator.cc",
        "torus_2d_reducescatter_translator.cc",
        "unidir_ring_reducescatter_translator.cc",
    ],
    hdrs = [
        "bidir_ring_reducescatter_translator.h",
        "mesh_1d_reducescatter_translator.h",
        "mesh_2d_reducescatter_translator.h",
        "ring_over_2d_grid_reducescatter_translator.h",
        "torus_2d_reducescatter_translator.h",
        "unidir_ring_reducescatter_translator.h",
    ],
    deps = [
        ":reducescatter_interface",
        "//paragraph/translation:factory",
        "//paragraph/translation:translation_utils",
        "//paragraph/translation/barrier:barrier",
        "//paragraph/translation/barrier:barrier_interface",
        "@libfactory//:factory",
        "@nlohmann_json//:nlohmann_json",
    ],
    visibility = [
        "//paragraph/translation:__subpackages__",
    ],
    alwayslink = 1,
)

cc_test(
    name = "reducescatter_test",
    size = "small",
    srcs = [
        "bidir_ring_reducescatter_translator_test.cc",
        "mesh_1d_reducescatter_translator_test.cc",
        "mesh_2d_reducescatter_translator_test.cc",
        "ring_over_2d_grid_reducescatter_translator_test.cc",
        "torus_2d_reducescatter_translator_test.cc",
        "unidir_ring_reducescatter_translator_test.cc",
    ],
    deps = [
        ":reducescatter",
        "//paragraph/shim:test_macros",
        "//paragraph/translation:factory",
        "@com_google_protobuf//:protobuf",
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
