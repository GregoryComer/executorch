#pragma once

#include <executorch/backends/xnnpack/runtime/graph/node.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>
#include <executorch/backends/xnnpack/runtime/operators/operator.h>
#include <executorch/runtime/core/span.h>

#include <memory>

namespace executorch::backends::xnnpack::operators::kleidi {

/*
 * Constructs the in-tree int4 GEMM operator for a Linear call node, or returns
 * nullptr if the node is not a dynamically-quantized int4 linear for which a
 * KleidiAI ukernel is available (e.g. KleidiAI is not compiled in, the CPU
 * lacks the required features, or the quant scheme/shape is unsupported).
 *
 * `input_specs` are the specs of the node's non-null inputs in argument order:
 * [activation (qdint8 dynamic), weight (blockwise int4), bias? (f32)].
 */
std::unique_ptr<Operator> make_linear_int4(
    const graph::CallOperatorNode& node,
    runtime::Span<const graph::TensorSpec> input_specs);

} // namespace executorch::backends::xnnpack::operators::kleidi
