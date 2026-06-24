#include <executorch/backends/xnnpack/runtime/plan/layout_assignment.h>

#include <executorch/backends/xnnpack/runtime/core/variant_util.h>
#include <executorch/runtime/platform/log.h>

#include <optional>
#include <vector>

namespace executorch::backends::xnnpack::plan {

using executorch::runtime::Span;
using graph::CallOperatorNode;
using graph::Graph;
using graph::NodeHandle;
using graph::TensorSpec;
using graph::ValueHandle;

namespace {

// The non-null input args of a call node, with their resolved specs.
struct Inputs {
  std::vector<ValueHandle> args;
  std::vector<TensorSpec> specs;
};

Inputs collect_inputs(const CallOperatorNode& node, const Graph& graph) {
  Inputs in;
  for (const auto& arg : node.args) {
    if (arg.is_null()) {
      continue;
    }
    in.args.push_back(arg);
    in.specs.push_back(graph.get_tensor_spec(arg));
  }
  return in;
}

// Sets the layout of the value produced at `vh`, which must be an operator
// output. Errors on a conflicting assignment or a non-operator producer.
runtime::Error
set_producer_layout(Graph& graph, ValueHandle vh, const core::Layout& layout) {
  auto* producer = std::get_if<CallOperatorNode>(&graph.nodes[vh.node].value);
  ET_CHECK_OR_RETURN_ERROR(
      producer != nullptr,
      NotSupported,
      "A Fixed layout was required on a value not produced by an operator");

  auto assign = [&](TensorSpec& spec) -> runtime::Error {
    ET_CHECK_OR_RETURN_ERROR(
        !spec.layout.has_value() || *spec.layout == layout,
        NotSupported,
        "Conflicting layout requirements on a shared value");
    spec.layout = layout;
    return runtime::Error::Ok;
  };

  return std::visit(
      overloaded{
          [&](TensorSpec& s) { return assign(s); },
          [&](std::vector<TensorSpec>& v) -> runtime::Error {
            ET_CHECK_OR_RETURN_ERROR(
                vh.output < v.size(), Internal, "Output index out of range");
            return assign(v[vh.output]);
          }},
      producer->output_specs);
}

// The current layouts of a node's non-null inputs / its outputs.
std::vector<std::optional<core::Layout>> input_layouts(const Inputs& in) {
  std::vector<std::optional<core::Layout>> out;
  out.reserve(in.specs.size());
  for (const auto& s : in.specs) {
    out.push_back(s.layout);
  }
  return out;
}

std::vector<std::optional<core::Layout>> output_layouts(
    const CallOperatorNode& node) {
  std::vector<std::optional<core::Layout>> out;
  std::visit(
      overloaded{
          [&](const TensorSpec& s) { out.push_back(s.layout); },
          [&](const std::vector<TensorSpec>& v) {
            for (const auto& s : v) {
              out.push_back(s.layout);
            }
          }},
      node.output_specs);
  return out;
}

} // namespace

runtime::Result<OperatorMap> instantiate_operators(const Graph& graph) {
  OperatorMap ops;
  for (NodeHandle h = 0; h < graph.nodes.size(); h++) {
    auto* node = std::get_if<CallOperatorNode>(&graph.nodes[h].value);
    if (node == nullptr) {
      continue;
    }
    auto in = collect_inputs(*node, graph);
    auto op =
        operators::create_operator(*node, {in.specs.data(), in.specs.size()});
    ET_CHECK_OR_RETURN_ERROR(
        op != nullptr,
        NotSupported,
        "No in-tree kernel for operator %d",
        static_cast<int>(node->op));
    ops.emplace(h, std::move(op));
  }
  return ops;
}

runtime::Error propagate_layouts(Graph& graph, const OperatorMap& ops) {
  // Push each operator's Fixed input requirements onto its producers.
  for (const auto& [handle, op] : ops) {
    const auto* node =
        std::get_if<CallOperatorNode>(&graph.nodes[handle].value);
    if (node == nullptr) {
      continue;
    }
    auto in = collect_inputs(*node, graph);
    auto reqs = op->required_input_layouts({in.specs.data(), in.specs.size()});
    for (size_t i = 0; i < reqs.size() && i < in.args.size(); i++) {
      if (reqs[i].kind != operators::LayoutConstraint::Kind::Fixed) {
        continue;
      }
      ET_CHECK_OR_RETURN_ERROR(
          reqs[i].layout.has_value(),
          Internal,
          "Fixed layout constraint without a layout");
      ET_CHECK_OK_OR_RETURN_ERROR(
          set_producer_layout(graph, in.args[i], *reqs[i].layout));
    }
  }

  // Now that all layouts are assigned, tell each operator the layouts that were
  // resolved for its inputs and outputs (flexible producers act on this).
  for (const auto& [handle, op] : ops) {
    const auto* node =
        std::get_if<CallOperatorNode>(&graph.nodes[handle].value);
    if (node == nullptr) {
      continue;
    }
    auto in = collect_inputs(*node, graph);
    auto ins = input_layouts(in);
    auto outs = output_layouts(*node);
    op->configure_layouts({ins.data(), ins.size()}, {outs.data(), outs.size()});
  }

  return runtime::Error::Ok;
}

} // namespace executorch::backends::xnnpack::plan
