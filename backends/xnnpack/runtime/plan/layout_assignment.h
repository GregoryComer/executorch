#pragma once

#include <executorch/backends/xnnpack/runtime/graph/graph.h>
#include <executorch/backends/xnnpack/runtime/graph/handles.h>
#include <executorch/backends/xnnpack/runtime/operators/operator.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#include <map>
#include <memory>

namespace executorch::backends::xnnpack::plan {

// In-tree operators, keyed by the handle of the call node they implement.
// Ordered for deterministic iteration.
using OperatorMap =
    std::map<graph::NodeHandle, std::unique_ptr<operators::Operator>>;

/*
 * Instantiate an in-tree operator for every call node not delegated to XNNPACK.
 * Errors if such a node has no in-tree kernel. Must run after partitioning (so
 * delegated nodes have been fused away).
 */
runtime::Result<OperatorMap> instantiate_operators(const graph::Graph& graph);

/*
 * Assign physical layouts to values so that each in-tree operator's input
 * layout requirements are satisfied by the layout its producer emits, then
 * inform every in-tree operator of its resolved input/output layouts (so
 * flexible producers learn the layout they must emit).
 *
 * v1 assigns only Fixed input requirements (e.g. a Kleidi-packed LHS), which
 * must target operator-produced values; a conflicting requirement on a shared
 * value is reported as an error rather than resolved with a relayout op.
 */
runtime::Error propagate_layouts(graph::Graph& graph, const OperatorMap& ops);

} // namespace executorch::backends::xnnpack::plan
