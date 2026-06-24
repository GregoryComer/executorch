#include <gtest/gtest.h>

#include <executorch/backends/xnnpack/runtime/core/layout.h>
#include <executorch/backends/xnnpack/runtime/core/tensor.h>
#include <executorch/backends/xnnpack/runtime/graph/graph_builder.h>
#include <executorch/backends/xnnpack/runtime/operators/operator.h>
#include <executorch/backends/xnnpack/runtime/plan/layout_assignment.h>

#include <memory>
#include <optional>
#include <vector>

using namespace executorch::backends::xnnpack::core;
using namespace executorch::backends::xnnpack::graph;
using namespace executorch::backends::xnnpack::plan;
using executorch::runtime::Error;
using executorch::runtime::Span;

namespace ops = executorch::backends::xnnpack::operators;

namespace {

// Requires a fixed layout on its first input; agnostic about the rest.
struct FixedInputConsumer : ops::Operator {
  Layout required;
  explicit FixedInputConsumer(Layout l) : required(std::move(l)) {}
  void execute(Span<Tensor*>, Span<Tensor*>) override {}
  std::vector<ops::LayoutConstraint> required_input_layouts(
      Span<const TensorSpec> inputs) const override {
    std::vector<ops::LayoutConstraint> r;
    r.push_back(ops::LayoutConstraint::fixed(required));
    for (size_t i = 1; i < inputs.size(); i++) {
      r.push_back(ops::LayoutConstraint::any());
    }
    return r;
  }
};

// Flexible producer: records the output layouts the pass resolves for it.
struct RecordingProducer : ops::Operator {
  bool configured = false;
  std::vector<std::optional<Layout>> outputs;
  void execute(Span<Tensor*>, Span<Tensor*>) override {}
  void configure_layouts(
      Span<const std::optional<Layout>>,
      Span<const std::optional<Layout>> resolved_outputs) override {
    configured = true;
    outputs.assign(resolved_outputs.begin(), resolved_outputs.end());
  }
};

Layout opaque(uint32_t scheme) {
  OpaquePacked op;
  op.scheme_id = scheme;
  return op;
}

ValueHandle make_int4_weight(GraphBuilder& b) {
  auto t = std::make_shared<Tensor>();
  t->dtype = DType::QInt4;
  t->sizes = {32, 64};
  return b.createConstant(t, qint4_blockwise_sym(1, 32));
}

// A Quantize producer feeding a Linear consumer (+ an int4 weight constant).
struct DqLinearGraph {
  Graph graph;
  ValueHandle producer;
  ValueHandle consumer;
};

DqLinearGraph build_dq_linear() {
  GraphBuilder b;
  auto fin = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(4), DimSizeSpec::constant(64)}};
  auto input = b.createInput(fin);
  auto qspec = TensorSpec{
      .dtype = DType::QInt8,
      .sizes = {DimSizeSpec::constant(4), DimSizeSpec::constant(64)},
      .quant_params = PerRowQuantParams{.axis = -1, .is_dynamic = true}};
  auto producer = b.createOperator(Operator::Quantize, qspec, input);
  auto weight = make_int4_weight(b);
  auto out = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(4), DimSizeSpec::constant(32)}};
  auto consumer =
      b.createOperator(Operator::Linear, out, ValueHandles{producer, weight});
  b.createOutput(consumer);

  DqLinearGraph g{b.build(), producer, consumer};
  g.graph.update_users();
  return g;
}

const TensorSpec& producer_output_spec(const Graph& graph, ValueHandle vh) {
  const auto* node = std::get_if<CallOperatorNode>(&graph.nodes[vh.node].value);
  return std::get<TensorSpec>(node->output_specs);
}

} // namespace

TEST(TestLayoutAssignment, fixed_input_propagates_to_producer) {
  auto g = build_dq_linear();
  auto layout = opaque(7);

  auto producer_op = std::make_unique<RecordingProducer>();
  RecordingProducer* producer_raw = producer_op.get();

  OperatorMap ops;
  ops.emplace(g.producer.node, std::move(producer_op));
  ops.emplace(g.consumer.node, std::make_unique<FixedInputConsumer>(layout));

  ASSERT_EQ(propagate_layouts(g.graph, ops), Error::Ok);

  // The producer's output value now carries the consumer's required layout.
  const auto& spec = producer_output_spec(g.graph, g.producer);
  ASSERT_TRUE(spec.layout.has_value());
  EXPECT_EQ(*spec.layout, layout);

  // The flexible producer was told the layout it must emit.
  EXPECT_TRUE(producer_raw->configured);
  ASSERT_EQ(producer_raw->outputs.size(), 1u);
  ASSERT_TRUE(producer_raw->outputs[0].has_value());
  EXPECT_EQ(*producer_raw->outputs[0], layout);
}

TEST(TestLayoutAssignment, conflicting_requirements_error) {
  // Two consumers of the same producer demanding different layouts.
  GraphBuilder b;
  auto fin = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(4), DimSizeSpec::constant(64)}};
  auto input = b.createInput(fin);
  auto qspec = TensorSpec{
      .dtype = DType::QInt8,
      .sizes = {DimSizeSpec::constant(4), DimSizeSpec::constant(64)},
      .quant_params = PerRowQuantParams{.axis = -1, .is_dynamic = true}};
  auto producer = b.createOperator(Operator::Quantize, qspec, input);
  auto w1 = make_int4_weight(b);
  auto w2 = make_int4_weight(b);
  auto out = TensorSpec{
      .dtype = DType::Float32,
      .sizes = {DimSizeSpec::constant(4), DimSizeSpec::constant(32)}};
  auto c1 = b.createOperator(Operator::Linear, out, ValueHandles{producer, w1});
  auto c2 = b.createOperator(Operator::Linear, out, ValueHandles{producer, w2});
  b.createOutput(c1);
  b.createOutput(c2);
  auto graph = b.build();
  graph.update_users();

  OperatorMap ops;
  ops.emplace(producer.node, std::make_unique<RecordingProducer>());
  ops.emplace(c1.node, std::make_unique<FixedInputConsumer>(opaque(1)));
  ops.emplace(c2.node, std::make_unique<FixedInputConsumer>(opaque(2)));

  EXPECT_NE(propagate_layouts(graph, ops), Error::Ok);
}
