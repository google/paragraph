package(licenses = ["notice"])

cc_library(
    name = "allreduce_interface",
    srcs = [
        "allreduce_translator.cc",
    ],
    hdrs = [
        "allreduce_translator.h",
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
    name = "allreduce",
    srcs = [
        "bidir_ring_allreduce_translator.cc",
        "dissemination_allreduce_translator.cc",
        "mesh_1d_allreduce_translator.cc",
        "mesh_2d_allreduce_translator.cc",
        "ring_over_2d_grid_allreduce_translator.cc",
        "torus_2d_allreduce_translator.cc",
        "unidir_ring_allreduce_translator.cc",
    ],
    hdrs = [
        "bidir_ring_allreduce_translator.h",
        "dissemination_allreduce_translator.h",
        "mesh_1d_allreduce_translator.h",
        "mesh_2d_allreduce_translator.h",
        "ring_over_2d_grid_allreduce_translator.h",
        "torus_2d_allreduce_translator.h",
        "unidir_ring_allreduce_translator.h",
    ],
    deps = [
        ":allreduce_interface",
        "//paragraph/translation:factory",
        "//paragraph/translation:translation_utils",
        "//paragraph/translation/allgather:allgather",
        "//paragraph/translation/allgather:allgather_interface",
        "//paragraph/translation/barrier:barrier",
        "//paragraph/translation/barrier:barrier_interface",
        "//paragraph/translation/reducescatter:reducescatter",
        "//paragraph/translation/reducescatter:reducescatter_interface",
        "@libfactory//:factory",
        "@nlohmann_json//:nlohmann_json",
    ],
    visibility = [
        "//paragraph/translation:__subpackages__",
    ],
    alwayslink = 1,
)

cc_test(
    name = "allreduce_test",
    size = "small",
    srcs = [
        "bidir_ring_allreduce_translator_test.cc",
        "dissemination_allreduce_translator_test.cc",
        "mesh_1d_allreduce_translator_test.cc",
        "mesh_2d_allreduce_translator_test.cc",
        "ring_over_2d_grid_allreduce_translator_test.cc",
        "torus_2d_allreduce_translator_test.cc",
        "unidir_ring_allreduce_translator_test.cc",
    ],
    deps = [
        ":allreduce",
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
