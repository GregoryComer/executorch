#include <executorch/backends/xnnpack/runtime/plan/xnn_support.h>

#include <executorch/backends/xnnpack/runtime/core/dtype.h>
#include <executorch/backends/xnnpack/runtime/core/quant_params.h>
#include <executorch/backends/xnnpack/runtime/core/variant_util.h>
#include <executorch/backends/xnnpack/runtime/operators/kleidi/kai_ukernel.h>
#include <executorch/runtime/platform/log.h>

#include <optional>

namespace executorch::backends::xnnpack::plan {

namespace {

using namespace graph;

bool check_xnn_dtype_support(core::DType dtype) {
  switch (dtype) {
    case core::DType::Float32:
    case core::DType::Float16:
    case core::DType::QUInt8:
    case core::DType::QInt8:
    case core::DType::QInt32:
    case core::DType::QInt4:
      return true;
    default:
      return false;
  }
}

bool check_xnn_op_support(Operator op) {
  switch (op) {
    case Operator::Add:
    case Operator::Subtract:
    case Operator::Multiply:
    case Operator::Divide:
    case Operator::Maximum:
    case Operator::Minimum:
    case Operator::CopySign:
    case Operator::SquaredDifference:
    case Operator::PReLU:
    case Operator::Modulus:
    case Operator::Atan2:
    case Operator::Pow:
    case Operator::Abs:
    case Operator::Negate:
    case Operator::Clamp:
    case Operator::Ceiling:
    case Operator::Floor:
    case Operator::Round:
    case Operator::Square:
    case Operator::SquareRoot:
    case Operator::ReciprocalSquareRoot:
    case Operator::Exp:
    case Operator::Log:
    case Operator::Sigmoid:
    case Operator::Tanh:
    case Operator::ELU:
    case Operator::GELU:
    case Operator::HardSwish:
    case Operator::LeakyReLU:
    case Operator::Sine:
    case Operator::Cosine:
    case Operator::Sign:
    case Operator::ReLU:
    case Operator::Linear:
    case Operator::BatchMatrixMultiply:
    case Operator::Conv2d:
    case Operator::ConvTranspose2d:
    case Operator::DepthwiseConv2d:
    case Operator::AvgPool2d:
    case Operator::AdaptiveAvgPool2d:
    case Operator::MaxPool2d:
    case Operator::Softmax:
    case Operator::Mean:
    case Operator::Sum:
    case Operator::Reshape:
    case Operator::View:
    case Operator::Transpose:
    case Operator::Permute:
    case Operator::Slice:
    case Operator::Cat:
    case Operator::Unsqueeze:
    case Operator::Expand:
    case Operator::Clone:
    case Operator::Pad:
    case Operator::StaticResizeBilinear2D:
    case Operator::Quantize:
    case Operator::Dequantize:
      return true;
    default:
      return false;
  }
}

} // namespace

bool check_xnn_node_support(const CallOperatorNode& node, const Graph& graph) {
  if (!check_xnn_op_support(node.op)) {
    return false;
  }

  for (auto& arg : node.args) {
    if (arg.is_null())
      continue;
    const auto& tensor_spec = graph.get_tensor_spec(arg);

    if (!check_xnn_dtype_support(tensor_spec.dtype)) {
      return false;
    }
  }

  return true;
}

namespace {

// Returns the constant extent of a tensor dim, or nullopt if it is symbolic.
std::optional<uint64_t> const_dim(const graph::TensorSpec& spec, size_t i) {
  if (i >= spec.sizes.size() || !spec.sizes[i].is_constant()) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(spec.sizes[i].offset);
}

// True if `spec` is a dynamically-quantized per-row int8 activation (qdint8).
bool is_dynamic_qint8(const graph::TensorSpec& spec) {
  if (spec.dtype != core::DType::QInt8 || !spec.quant_params) {
    return false;
  }
  auto* pr = std::get_if<core::PerRowQuantParams>(&*spec.quant_params);
  return pr != nullptr && pr->is_dynamic;
}

// True if the Linear `node` is an int4 dynamically-quantized linear for which
// an in-tree Kleidi ukernel is available.
bool is_int4_dynamic_linear(const CallOperatorNode& node, const Graph& graph) {
  if (node.op != Operator::Linear || node.args.size() < 2 ||
      node.args[0].is_null() || node.args[1].is_null()) {
    return false;
  }
  auto act = graph.get_tensor_spec(node.args[0]);
  auto weight = graph.get_tensor_spec(node.args[1]);
  if (!is_dynamic_qint8(act) || weight.dtype != core::DType::QInt4 ||
      !weight.quant_params) {
    return false;
  }
  auto* pb = std::get_if<core::PerBlockQuantParams>(&*weight.quant_params);
  if (pb == nullptr) {
    return false;
  }
  // Weight is a constant [n, k] matrix.
  auto n = const_dim(weight, 0);
  auto k = const_dim(weight, 1);
  if (!n || !k) {
    return false;
  }
  return operators::kleidi::select_qsi4c32p_ukernel(
             *n, *k, static_cast<uint32_t>(pb->block_size))
      .has_value();
}

// True if the dynamic-quant node at `handle` produces the activation for an
// in-tree int4 linear. Requires *all* users to be such linears, so we never
// hand a Kleidi-packed buffer to an op that cannot read it.
bool feeds_only_int4_linears(NodeHandle handle, const Graph& graph) {
  const auto& users = graph.nodes[handle].users;
  if (users.empty()) {
    return false;
  }
  for (auto user : users) {
    auto* op = std::get_if<CallOperatorNode>(&graph.nodes[user].value);
    if (op == nullptr || !is_int4_dynamic_linear(*op, graph) ||
        op->args[0].node != handle) {
      return false;
    }
  }
  return true;
}

} // namespace

bool prefer_in_tree_kernel(NodeHandle handle, const Graph& graph) {
  auto* node = std::get_if<CallOperatorNode>(&graph.nodes[handle].value);
  if (node == nullptr) {
    return false;
  }

  // The int4 GEMM itself.
  if (is_int4_dynamic_linear(*node, graph)) {
    return true;
  }

  // The dynamic-quant (convert -> qdint8) that feeds an int4 GEMM: it becomes
  // the in-tree LHS-pack op, so it must be kept out of the XNNPACK subgraph
  // too.
  if (node->op == Operator::Quantize) {
    auto* out = std::get_if<graph::TensorSpec>(&node->output_specs);
    if (out != nullptr && is_dynamic_qint8(*out) &&
        feeds_only_int4_linears(handle, graph)) {
      return true;
    }
  }

  return false;
}

} // namespace executorch::backends::xnnpack::plan
