#include <executorch/backends/xnnpack/runtime/operators/kleidi/dynamic_quant_pack.h>

#include <executorch/backends/xnnpack/runtime/core/layout.h>
#include <executorch/backends/xnnpack/runtime/core/quant_params.h>
#include <executorch/backends/xnnpack/runtime/operators/kleidi/kai_ukernel.h>
#include <executorch/runtime/platform/assert.h>

#include <optional>
#include <variant>
#include <vector>

namespace executorch::backends::xnnpack::operators::kleidi {

using core::DType;
using core::Layout;
using core::OpaquePacked;
using core::PerRowQuantParams;
using executorch::runtime::Span;
using graph::CallOperatorNode;
using graph::TensorSpec;

namespace {

bool produces_dynamic_qint8(const CallOperatorNode& node) {
  const auto* out = std::get_if<TensorSpec>(&node.output_specs);
  if (out == nullptr || out->dtype != DType::QInt8 || !out->quant_params) {
    return false;
  }
  auto* pr = std::get_if<PerRowQuantParams>(&*out->quant_params);
  return pr != nullptr && pr->is_dynamic;
}

// Quantizes an fp32 activation to per-row int8 and packs it into the Kleidi LHS
// layout its consumer GEMM requires. A flexible producer: the required layout
// is learned from the propagation pass via configure_layouts.
class DynamicQuantPack : public Operator {
 public:
  std::vector<ResultLayout> result_layouts(
      Span<const TensorSpec>) const override {
    // The output layout is dictated by the consuming GEMM.
    return {ResultLayout::inherit()};
  }

  void configure_layouts(
      Span<const std::optional<Layout>>,
      Span<const std::optional<Layout>> resolved_outputs) override {
    if (resolved_outputs.empty() || !resolved_outputs[0].has_value()) {
      return;
    }
    if (auto* op = std::get_if<OpaquePacked>(&*resolved_outputs[0])) {
      cfg_ = lhs_config_from_layout(*op);
    }
  }

  void execute(Span<core::Tensor*> inputs, Span<core::Tensor*> outputs)
      override {
    ET_CHECK(!inputs.empty() && !outputs.empty());
    ET_CHECK_MSG(
        cfg_.has_value(), "DynamicQuantPack ran without a resolved LHS layout");
    const auto* in = inputs[0];
    auto* out = outputs[0];

    size_t k = in->sizes.empty() ? 0 : static_cast<size_t>(in->sizes.back());
    size_t m = 1;
    for (size_t i = 0; i + 1 < in->sizes.size(); i++) {
      m *= static_cast<size_t>(in->sizes[i]);
    }

    run_lhs_quant_pack(
        *cfg_,
        m,
        k,
        in->data_const<float>(),
        /*lhs_stride=*/k * sizeof(float),
        out->storage.data);
  }

 private:
  std::optional<KaiUkernelConfig> cfg_;
};

} // namespace

std::unique_ptr<Operator> make_dynamic_quant_pack(
    const CallOperatorNode& node,
    Span<const TensorSpec> input_specs) {
  if (node.op != graph::Operator::Quantize || input_specs.empty() ||
      !produces_dynamic_qint8(node)) {
    return nullptr;
  }
  return std::make_unique<DynamicQuantPack>();
}

} // namespace executorch::backends::xnnpack::operators::kleidi
