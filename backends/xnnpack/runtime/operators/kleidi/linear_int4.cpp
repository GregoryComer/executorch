#include <executorch/backends/xnnpack/runtime/operators/kleidi/linear_int4.h>

#include <executorch/backends/xnnpack/runtime/core/quant_params.h>
#include <executorch/backends/xnnpack/runtime/operators/kleidi/kai_ukernel.h>
#include <executorch/runtime/platform/assert.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace executorch::backends::xnnpack::operators::kleidi {

using core::DType;
using core::PerBlockQuantParams;
using core::PerRowQuantParams;
using executorch::runtime::Span;
using graph::CallOperatorNode;
using graph::TensorSpec;

namespace {

bool is_dynamic_qint8(const TensorSpec& spec) {
  if (spec.dtype != DType::QInt8 || !spec.quant_params) {
    return false;
  }
  auto* pr = std::get_if<PerRowQuantParams>(&*spec.quant_params);
  return pr != nullptr && pr->is_dynamic;
}

std::optional<uint64_t> const_dim(const TensorSpec& spec, size_t i) {
  if (i >= spec.sizes.size() || !spec.sizes[i].is_constant()) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(spec.sizes[i].offset);
}

// Dynamically-quantized int8 activation x blockwise int4 weight -> f32. The
// weight is packed once in prepare(); the activation arrives already packed
// (the dynamic-quant op emits the Kleidi LHS layout) and the matmul runs in
// execute().
class LinearInt4 : public Operator {
 public:
  LinearInt4(
      KaiUkernelConfig cfg,
      uint64_t n,
      uint64_t k,
      bool has_bias,
      float output_min,
      float output_max)
      : cfg_(cfg),
        n_(n),
        k_(k),
        has_bias_(has_bias),
        output_min_(output_min),
        output_max_(output_max) {}

  std::vector<LayoutConstraint> required_input_layouts(
      Span<const TensorSpec> input_specs) const override {
    std::vector<LayoutConstraint> reqs;
    reqs.reserve(input_specs.size());
    // Activation must be the Kleidi-packed LHS; weight/bias are unconstrained
    // (the weight is packed internally in prepare()).
    reqs.push_back(LayoutConstraint::fixed(lhs_layout(cfg_)));
    for (size_t i = 1; i < input_specs.size(); i++) {
      reqs.push_back(LayoutConstraint::any());
    }
    return reqs;
  }

  void prepare(Span<core::Tensor*> inputs, Span<core::Tensor*> outputs)
      override {
    (void)outputs;
    ET_CHECK(inputs.size() >= 2);
    const auto* weight = inputs[1];
    ET_CHECK_MSG(
        !weight->aux_storage.empty(),
        "int4 weight is missing block scale data");

    const auto* rhs = weight->data_const<uint8_t>();
    const void* scales = weight->aux_storage[0].data;
    const float* bias = has_bias_ ? inputs[2]->data_const<float>() : nullptr;
    // bf16 scales, one per (k / block) block per output row.
    size_t scale_stride = (k_ / cfg_.bl) * sizeof(uint16_t);

    packed_rhs_.resize(rhs_packed_size(cfg_, n_, k_));
    run_rhs_pack(
        cfg_, n_, k_, rhs, bias, scales, scale_stride, packed_rhs_.data());
  }

  void execute(Span<core::Tensor*> inputs, Span<core::Tensor*> outputs)
      override {
    ET_CHECK(!inputs.empty() && !outputs.empty());
    const auto* lhs = inputs[0];
    auto* out = outputs[0];

    // Activation rows = product of all logical dims except the reduction dim.
    size_t m = 1;
    for (size_t i = 0; i + 1 < lhs->sizes.size(); i++) {
      m *= static_cast<size_t>(lhs->sizes[i]);
    }

    run_matmul(
        cfg_,
        m,
        n_,
        k_,
        lhs->storage.data,
        packed_rhs_.data(),
        out->data_mut<float>(),
        /*dst_stride_row=*/n_ * sizeof(float),
        output_min_,
        output_max_);
  }

 private:
  KaiUkernelConfig cfg_;
  uint64_t n_;
  uint64_t k_;
  bool has_bias_;
  float output_min_;
  float output_max_;
  std::vector<uint8_t> packed_rhs_;
};

} // namespace

std::unique_ptr<Operator> make_linear_int4(
    const CallOperatorNode& node,
    Span<const TensorSpec> input_specs) {
  if (node.op != graph::Operator::Linear || input_specs.size() < 2) {
    return nullptr;
  }
  const auto& act = input_specs[0];
  const auto& weight = input_specs[1];
  if (!is_dynamic_qint8(act) || weight.dtype != DType::QInt4 ||
      !weight.quant_params) {
    return nullptr;
  }
  auto* pb = std::get_if<PerBlockQuantParams>(&*weight.quant_params);
  if (pb == nullptr) {
    return nullptr;
  }
  auto n = const_dim(weight, 0);
  auto k = const_dim(weight, 1);
  if (!n || !k) {
    return nullptr;
  }
  auto cfg =
      select_qsi4c32p_ukernel(*n, *k, static_cast<uint32_t>(pb->block_size));
  if (!cfg) {
    return nullptr;
  }
  bool has_bias = node.args.size() > 2 && !node.args[2].is_null();
  return std::make_unique<LinearInt4>(
      *cfg, *n, *k, has_bias, node.output_min, node.output_max);
}

} // namespace executorch::backends::xnnpack::operators::kleidi
