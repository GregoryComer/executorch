#include <executorch/backends/xnnpack/runtime/operators/kleidi/linear_int4.h>

#include <executorch/backends/xnnpack/runtime/cache/packed_weight_cache.h>
#include <executorch/backends/xnnpack/runtime/core/quant_params.h>
#include <executorch/backends/xnnpack/runtime/operators/kleidi/kai_ukernel.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#include <cstdint>
#include <optional>
#include <string>
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

  runtime::Error prepare(
      Span<core::Tensor*> inputs,
      Span<core::Tensor*> outputs,
      cache::PackedWeightCache* weight_cache) override {
    (void)outputs;
    ET_CHECK_OR_RETURN_ERROR(
        inputs.size() >= 2,
        InvalidArgument,
        "LinearInt4 expects at least 2 inputs");
    const auto* weight = inputs[1];
    ET_CHECK_OR_RETURN_ERROR(
        !weight->aux_storage.empty(),
        InvalidArgument,
        "int4 weight is missing block scale data");

    const auto* rhs = weight->data_const<uint8_t>();
    const void* scales = weight->aux_storage[0].data;
    const float* bias = has_bias_ ? inputs[2]->data_const<float>() : nullptr;
    // bf16 scales, one per (k / block) block per output row.
    size_t scale_stride = (k_ / cfg_.bl) * sizeof(uint16_t);
    size_t packed_size = rhs_packed_size(cfg_, n_, k_);

    // Cached path: pack once per (weight, packing) and share the buffer.
    if (weight_cache != nullptr) {
      cache::PackedWeightKey key{
          weight_source_id(weight), qsi4c32p_packing_fingerprint(cfg_)};
      if (auto hit = weight_cache->lookup(key)) {
        packed_rhs_ = hit->data();
        return runtime::Error::Ok;
      }
      auto reserved = weight_cache->reserve(packed_size);
      if (reserved.error() != runtime::Error::Ok) {
        return reserved.error();
      }
      void* buf = reserved.get();
      run_rhs_pack(cfg_, n_, k_, rhs, bias, scales, scale_stride, buf);
      auto committed = weight_cache->commit(key, buf, packed_size);
      if (committed.error() != runtime::Error::Ok) {
        return committed.error();
      }
      packed_rhs_ = committed.get().data();
      return runtime::Error::Ok;
    }

    // Uncached fallback: pack into operator-owned storage.
    owned_packed_.resize(packed_size);
    run_rhs_pack(
        cfg_, n_, k_, rhs, bias, scales, scale_stride, owned_packed_.data());
    packed_rhs_ = owned_packed_.data();
    return runtime::Error::Ok;
  }

  runtime::Error execute(
      Span<core::Tensor*> inputs,
      Span<core::Tensor*> outputs) override {
    ET_CHECK_OR_RETURN_ERROR(
        !inputs.empty() && !outputs.empty(),
        InvalidArgument,
        "LinearInt4 requires inputs and outputs");
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
        packed_rhs_,
        out->data_mut<float>(),
        /*dst_stride_row=*/n_ * sizeof(float),
        output_min_,
        output_max_);
    return runtime::Error::Ok;
  }

  std::vector<size_t> consumed_constant_inputs() const override {
    // Weight (and bias) are packed into the RHS during prepare(); execute()
    // reads only the packed buffer + the activation, so the unpacked source can
    // be freed.
    std::vector<size_t> consumed{1};
    if (has_bias_) {
      consumed.push_back(2);
    }
    return consumed;
  }

 private:
  // Identity of the *unpacked* source for the cache key: the PTD named_key when
  // present, else a stable per-process fallback (the weight's data pointer) for
  // inline constants that carry no name.
  static std::string weight_source_id(const core::Tensor* weight) {
    if (!weight->source_key.empty()) {
      return "key:" + weight->source_key;
    }
    return "ptr:" +
        std::to_string(static_cast<unsigned long long>(
            reinterpret_cast<uintptr_t>(weight->data_const<uint8_t>())));
  }

  KaiUkernelConfig cfg_;
  uint64_t n_;
  uint64_t k_;
  bool has_bias_;
  float output_min_;
  float output_max_;
  // Borrowed view of the packed weight: into the cache, or into owned_packed_.
  const uint8_t* packed_rhs_ = nullptr;
  std::vector<uint8_t> owned_packed_;
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
