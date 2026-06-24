#include <executorch/backends/xnnpack/runtime/operators/operator.h>

#include <executorch/runtime/platform/log.h>

namespace executorch::backends::xnnpack::operators {

std::unique_ptr<Operator> create_operator(
    const graph::CallOperatorNode& node,
    runtime::Span<const graph::TensorSpec> input_specs) {
  (void)input_specs;
  // No in-tree operators are available yet; the graph runtime currently
  // supports only XNNPACK-delegated subgraphs. Reaching this point means a
  // node was routed to an in-tree kernel that has not been added. Return null
  // so the caller can fail cleanly rather than aborting.
  ET_LOG(
      Error,
      "No in-tree kernel for operator %d; only XNNPACK-delegated nodes are "
      "supported",
      static_cast<int>(node.op));
  return nullptr;
}

} // namespace executorch::backends::xnnpack::operators
