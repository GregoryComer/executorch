#pragma once

#include <executorch/backends/xnnpack/runtime/graph/graph.h>
#include <executorch/backends/xnnpack/runtime/graph/handles.h>

namespace executorch::backends::xnnpack::plan {

// Returns true if XNNPACK can run the given operator node.
bool check_xnn_node_support(
    const graph::CallOperatorNode& node,
    const graph::Graph& graph);

// Returns true if we have a preferred in-tree kernel for the node at `handle`,
// meaning it should not be delegated to XNNPACK even when XNNPACK supports it.
// Takes a handle (rather than just the call node) so it can inspect a node's
// users -- e.g. to route a dynamic-quant node in-tree only when it feeds an
// in-tree int4 linear.
bool prefer_in_tree_kernel(graph::NodeHandle handle, const graph::Graph& graph);

} // namespace executorch::backends::xnnpack::plan
