#pragma once

#include <executorch/backends/xnnpack/runtime/core/dtype.h>
#include <executorch/runtime/core/span.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

/*
 * Tensor layout: how a value's logical elements (described by its shape +
 * dtype + quant params) are physically arranged in memory, plus any inline
 * metadata that rides along.
 *
 * Layout is orthogonal to the quantization *scheme* (which stays in
 * QuantParams). The absence of a layout (std::optional<Layout> == nullopt)
 * means the default: row-major, contiguous, no inline metadata -- the only
 * behavior that existed before this type, so existing values are unaffected.
 *
 * Three structured tiers plus an opaque escape hatch:
 *   - DimOrder    (Tier 1): an axis permutation (NCHW vs NHWC, transposed GEMM
 *                 operands). Same byte count as contiguous; conversion is a
 *                 transpose. DEFINED BUT UNWIRED -- no kernel produces/consumes
 *                 it yet.
 *   - Blocked     (Tier 2): tiled/padded geometry (e.g. nChw8c, generic GEMM
 *                 panels). DEFINED BUT UNWIRED.
 *   - OpaquePacked(Tier 3): a kernel-private packed buffer whose internal
 *                 structure we do not model. We only know which scheme produced
 *                 it and how to size it. This is what the KleidiAI int4 kernels
 *                 use for their packed LHS/RHS buffers.
 *
 * The layout-propagation pass is responsible for synthesizing conversion ops
 * between mismatched layouts; that logic lives with the pass, not here, so this
 * header stays free of any graph/operator dependency.
 */

namespace executorch::backends::xnnpack::core {

/*
 * Tier 1: a permutation of axes, minor-to-major, following ExecuTorch's core
 * dim_order convention. Does not change the element count. Unwired in v1.
 */
struct DimOrder {
  std::vector<uint8_t> order;

  bool operator==(const DimOrder& o) const {
    return order == o.order;
  }
};

/*
 * Tier 2: per-axis tiling. `tiles[i]` is the tile extent along axis i (0 means
 * the axis is not tiled); the physical size rounds each tiled axis up to a
 * multiple of its tile. Unwired in v1.
 */
struct Blocked {
  std::vector<uint32_t> tiles;

  bool operator==(const Blocked& o) const {
    return tiles == o.tiles;
  }
};

/*
 * Tier 3: an opaque, kernel-private packed buffer.
 *
 * `scheme_id` identifies the producer/consumer family (e.g. a KleidiAI int4 LHS
 * vs RHS packing); `params` carries the scheme-specific geometry (for KleidiAI,
 * the ukernel's mr/nr/kr/sr and friends). `size_fn` computes the packed byte
 * size for a given logical shape -- it is a plain function pointer (not a
 * std::function) so Layout stays trivially copyable and equality-comparable,
 * and so this header takes no dependency on the module that defines the packing
 * (e.g. operators/kleidi). The producing op installs `size_fn` when it stamps
 * the layout.
 */
struct OpaquePacked {
  static constexpr size_t kMaxParams = 6;

  uint32_t scheme_id = 0;
  std::array<uint32_t, kMaxParams> params{};
  size_t (*size_fn)(
      const OpaquePacked& self,
      runtime::Span<const uint64_t> sizes,
      DType dtype) = nullptr;

  bool operator==(const OpaquePacked& o) const {
    return scheme_id == o.scheme_id && params == o.params &&
        size_fn == o.size_fn;
  }
};

using Layout = std::variant<DimOrder, Blocked, OpaquePacked>;

} // namespace executorch::backends::xnnpack::core
