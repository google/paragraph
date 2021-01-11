cc_library(
    name = "hlo_converter",
    srcs = [
        "hlo_compute_cost_analysis.cc",
        "hlo_converter.cc",
    ],
    hdrs = [
        "hlo_compute_cost_analysis.h",
        "hlo_converter.h",
    ],
    deps = [
        "//paragraph/graph:graph",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory:memory",
        "@com_google_absl//absl/strings:strings",
        "@org_tensorflow//tensorflow/compiler/xla/client:local_client",
        "@org_tensorflow//tensorflow/compiler/xla/tools:hlo_module_loader",
    ],
    visibility = [
        "//visibility:public",
    ],
)

cc_test(
    name = "hlo_converter_test",
    size = "small",
    srcs = [
        "hlo_compute_cost_analysis_test.cc",
        "hlo_converter_test.cc",
    ],
    deps = [
        ":hlo_converter",
        "@googletest//:gtest_main",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_builder",
    ],
)

cc_binary(
    name = "hlo_bridge",
    srcs = ["hlo_bridge.cc"],
    deps = [
        ":hlo_converter",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
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
