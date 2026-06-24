#include <gtest/gtest.h>

#include <executorch/backends/xnnpack/runtime/core/quant_params.h>
#include <executorch/backends/xnnpack/runtime/graph/node.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>
#include <executorch/backends/xnnpack/runtime/operators/kleidi/kai_available.h>
#include <executorch/backends/xnnpack/runtime/operators/kleidi/linear_int4.h>

#include <vector>

using namespace executorch::backends::xnnpack::core;
using namespace executorch::backends::xnnpack::graph;
namespace kleidi = executorch::backends::xnnpack::operators::kleidi;
using executorch::runtime::Span;

namespace {

TensorSpec dynamic_qint8(int64_t m, int64_t k) {
  return TensorSpec{
      .dtype = DType::QInt8,
      .sizes = {DimSizeSpec::constant(m), DimSizeSpec::constant(k)},
      .quant_params = PerRowQuantParams{.axis = -1, .is_dynamic = true}};
}

TensorSpec int4_weight(int64_t n, int64_t k, int32_t block) {
  return TensorSpec{
      .dtype = DType::QInt4,
      .sizes = {DimSizeSpec::constant(n), DimSizeSpec::constant(k)},
      .quant_params = qint4_blockwise_sym(1, block)};
}

CallOperatorNode linear_node(size_t num_args) {
  CallOperatorNode node;
  node.op = Operator::Linear;
  node.args.assign(num_args, ValueHandle{0, 0});
  return node;
}

} // namespace

TEST(TestLinearInt4, rejects_non_linear) {
  CallOperatorNode node;
  node.op = Operator::Add;
  std::vector<TensorSpec> specs = {
      dynamic_qint8(4, 64), int4_weight(32, 64, 32)};
  EXPECT_EQ(
      kleidi::make_linear_int4(node, {specs.data(), specs.size()}), nullptr);
}

TEST(TestLinearInt4, rejects_non_quantized_linear) {
  auto node = linear_node(2);
  auto f32 = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(4), DimSizeSpec::constant(64)}};
  std::vector<TensorSpec> specs = {f32, f32};
  EXPECT_EQ(
      kleidi::make_linear_int4(node, {specs.data(), specs.size()}), nullptr);
}

TEST(TestLinearInt4, int4_dynamic_linear_matches_availability) {
  auto node = linear_node(3); // act, weight, bias
  std::vector<TensorSpec> specs = {
      dynamic_qint8(4, 64), int4_weight(32, 64, 32)};
  auto op = kleidi::make_linear_int4(node, {specs.data(), specs.size()});
  if (!kleidi::kleidi_compiled_in()) {
    EXPECT_EQ(op, nullptr);
  }
  // On a Kleidi build with a supporting CPU, op is non-null; otherwise
  // selection fails and it is null. Either way construction must not crash.
}
