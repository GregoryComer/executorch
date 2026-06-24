load("@fbsource//xplat/executorch/backends/xnnpack/third-party:third_party_libs.bzl", "third_party_dep")
load("@fbsource//xplat/executorch/build:build_variables.bzl", "XNNPACK_BACKEND_BUCK_SRCS")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def _get_preprocessor_flags():
    """
    Disable if someone explictly specified a config option,
    else Enable otherwise
    """
    preprocessor_flags = []
    if native.read_config("executorch", "xnnpack_workspace_sharing", "0") != "0":
        preprocessor_flags.append("-DENABLE_XNNPACK_SHARED_WORKSPACE")

    if native.read_config("executorch", "xnnpack_weights_cache", "0") != "0":
        preprocessor_flags.append("-DENABLE_XNNPACK_WEIGHTS_CACHE")

    # In-tree Kleidi kernels (and the legacy delegate's qp8 pack-for-gemm path)
    # are gated on ENABLE_XNNPACK_KLEIDI. Off by default; enabling it requires
    # the kleidiai dep below. ARM-only.
    if native.read_config("executorch", "xnnpack_kleidi", "0") != "0":
        preprocessor_flags.append("-DENABLE_XNNPACK_KLEIDI")

    preprocessor_flags.append("-DXNNPACK_WORKSPACE_ALWAYS_LOCK")

    # Enable if not disabled through config
    return preprocessor_flags

def _kleidi_deps():
    if native.read_config("executorch", "xnnpack_kleidi", "0") != "0":
        return [third_party_dep("kleidiai")]
    return []

def define_common_targets():
    runtime.cxx_library(
        name = "dynamic_quant_utils",
        srcs = [
            "runtime/utils/utils.cpp",
        ],
        exported_headers = ["runtime/utils/utils.h"],
        deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/backend:interface",
        ],
        visibility = ["PUBLIC"],
    )

    for aten_mode in get_aten_mode_options():
        aten_suffix = "_aten" if aten_mode else ""
        runtime.cxx_library(
            name = "xnnpack_backend" + aten_suffix,
            srcs = XNNPACK_BACKEND_BUCK_SRCS,
            headers = native.glob([
                "runtime/*.h",
                "runtime/profiling/*.h",
                "runtime/core/*.h",
                "runtime/graph/*.h",
                "runtime/operators/*.h",
                "runtime/operators/kleidi/*.h",
                "runtime/executor/*.h",
                "runtime/plan/*.h",
            ]),
            visibility = ["PUBLIC"],
            preprocessor_flags = [
                # Uncomment to enable per operator timings
                # "-DENABLE_XNNPACK_PROFILING",
                # KleidiAI is enabled via `-c executorch.xnnpack_kleidi=1`
                # (see _get_preprocessor_flags / _kleidi_deps).
            ] + _get_preprocessor_flags(),
            exported_deps = [
                "//executorch/runtime/backend:interface" + aten_suffix,
            ],
            exported_headers = [
                "runtime/XNNPACKBackend.h",
            ],
            deps = [
                third_party_dep("XNNPACK"),
                "//executorch/backends/xnnpack/serialization:xnnpack_flatbuffer_header",
                "//executorch/extension/threadpool:threadpool",
                "//executorch/runtime/core/exec_aten/util:tensor_util" + aten_suffix,
                "//executorch/runtime/executor:pte_data_map" + aten_suffix,
            ] + _kleidi_deps(),
            # XnnpackBackend.cpp needs to compile with executor as whole
            # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
            link_whole = True,
        )
    
    runtime.cxx_library(
        name = "xnnpack_interface",
        visibility = ["PUBLIC"],
        exported_headers = [
            "runtime/XNNPACKBackend.h",
        ],
    )
