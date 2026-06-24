#pragma once

#include <executorch/backends/xnnpack/runtime/core/dtype.h>
#include <executorch/backends/xnnpack/runtime/core/layout.h>

#include <cstddef>
#include <cstdint>
#include <optional>

/*
 * Thin C++ wrapper over the KleidiAI matmul_clamp_f32_qai8dxp_qsi4c32p ukernel
 * family (dynamically-quantized int8 activations x blockwise int4 weights).
 *
 * This header is always compilable; it does not include any kai/ headers. When
 * KleidiAI is not compiled in (see kai_available.h), selection returns nullopt
 * and the run wrappers are unreachable. All knowledge of concrete kai ukernel
 * variants and their interface structs lives in kai_ukernel.cpp, keyed by
 * KaiUkernelConfig::variant_id, so the rest of the backend never sees a kai
 * type.
 */

namespace executorch::backends::xnnpack::operators::kleidi {

/*
 * OpaquePacked::scheme_id values for the layouts this module produces. Stable;
 * stored in serialized-independent runtime layouts only.
 */
enum KaiScheme : uint32_t {
  kSchemeNone = 0,
  kSchemeQai8dxpLhs = 1, // dynamically-quantized int8 packed activation (LHS)
  kSchemeQsi4c32pRhs = 2, // blockwise int4 packed weights (RHS)
};

/*
 * A selected matmul ukernel variant and the geometry callers need. The concrete
 * kai interface struct is looked up from `variant_id` inside kai_ukernel.cpp.
 */
struct KaiUkernelConfig {
  uint32_t variant_id = 0;
  uint32_t mr = 0;
  uint32_t nr = 0;
  uint32_t kr = 0;
  uint32_t sr = 0;
  uint32_t bl = 0; // block length (group size) for qsi4c32p weights
};

/*
 * Select a matmul ukernel for an int4 dynamically-quantized linear with weight
 * shape [n, k] and the given block size, based on the host's CPU features.
 * Returns nullopt if KleidiAI is unavailable or no variant fits the problem
 * (e.g. block size not a multiple of 32, or k not a multiple of block size).
 */
std::optional<KaiUkernelConfig>
select_qsi4c32p_ukernel(uint64_t n, uint64_t k, uint32_t block_size);

/*
 * Build the OpaquePacked layouts for a selected ukernel's packed LHS and RHS
 * buffers. The returned layout carries the geometry + a size hook so memory
 * planning can size the packed buffer from a logical shape.
 */
core::OpaquePacked lhs_layout(const KaiUkernelConfig& cfg);
core::OpaquePacked rhs_layout(const KaiUkernelConfig& cfg);

/*
 * Runtime wrappers used by the in-tree ops. m is the activation row count
 * (tokens), k the reduction dim, n the output channel count. These forward to
 * the corresponding kai_run_* / kai_get_*_size functions for the selected
 * variant. Unreachable (and abort) when KleidiAI is not compiled in.
 */
size_t lhs_packed_size(const KaiUkernelConfig& cfg, size_t m, size_t k);
void run_lhs_quant_pack(
    const KaiUkernelConfig& cfg,
    size_t m,
    size_t k,
    const float* lhs,
    size_t lhs_stride,
    void* lhs_packed);

size_t rhs_packed_size(const KaiUkernelConfig& cfg, size_t n, size_t k);
void run_rhs_pack(
    const KaiUkernelConfig& cfg,
    size_t n,
    size_t k,
    const uint8_t* rhs,
    const float* bias,
    const void* scale,
    size_t scale_stride,
    void* rhs_packed);

void run_matmul(
    const KaiUkernelConfig& cfg,
    size_t m,
    size_t n,
    size_t k,
    const void* lhs_packed,
    const void* rhs_packed,
    float* dst,
    size_t dst_stride_row,
    float clamp_min,
    float clamp_max);

} // namespace executorch::backends::xnnpack::operators::kleidi
