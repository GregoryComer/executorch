#pragma once

#include <executorch/backends/xnnpack/runtime/graph/node.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>
#include <executorch/backends/xnnpack/runtime/operators/operator.h>
#include <executorch/runtime/core/span.h>

#include <memory>

namespace executorch::backends::xnnpack::operators::kleidi {

/*
 * Constructs the in-tree dynamic-quant LHS-pack operator for a Quantize node
 * that produces a dynamically-quantized int8 activation (qdint8), or nullptr if
 * the node is not such a convert. The op is a flexible producer: the layout-
 * propagation pass tells it (via configure_layouts) which Kleidi-packed LHS
 * layout its consumer GEMM requires, and execute() quantizes + packs the
 * activation into that layout in a single step.
 *
 * Partitioning only routes a Quantize node in-tree when it feeds int4 linears,
 * so reaching this factory implies the LHS-pack role.
 */
std::unique_ptr<Operator> make_dynamic_quant_pack(
    const graph::CallOperatorNode& node,
    runtime::Span<const graph::TensorSpec> input_specs);

} // namespace executorch::backends::xnnpack::operators::kleidi
