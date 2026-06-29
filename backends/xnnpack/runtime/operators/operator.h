#pragma once

#include <executorch/backends/xnnpack/runtime/core/layout.h>
#include <executorch/backends/xnnpack/runtime/core/tensor.h>
#include <executorch/backends/xnnpack/runtime/graph/node.h>
#include <executorch/backends/xnnpack/runtime/graph/operator.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/span.h>

#include <memory>
#include <optional>
#include <vector>

namespace executorch::backends::xnnpack::operators {

/*
 * What layout an operator requires of one of its inputs.
 *   Any        - layout-agnostic; the input may be in any layout (the
 *                propagation pass is free to choose / leave it as produced).
 *   Contiguous - the input must be in the default row-major layout.
 *   Fixed      - the input must be in exactly `layout` (e.g. a Kleidi packed
 *                buffer); the pass inserts a relayout op if the producer
 * differs.
 */
struct LayoutConstraint {
  enum class Kind { Any, Contiguous, Fixed };
  Kind kind = Kind::Any;
  std::optional<core::Layout> layout;

  static LayoutConstraint any() {
    return {};
  }
  static LayoutConstraint contiguous() {
    return {Kind::Contiguous, std::nullopt};
  }
  static LayoutConstraint fixed(core::Layout l) {
    return {Kind::Fixed, std::move(l)};
  }
};

/*
 * What layout an operator produces for one of its results.
 *   Inherit - the op is a flexible producer; the propagation pass decides the
 *             result layout from the consumer(s) and reports it back via
 *             configure_layouts (e.g. the dynamic-quant op packing its output
 *             for whichever GEMM ukernel consumes it).
 *   Fixed   - the op always produces `layout`.
 */
struct ResultLayout {
  enum class Kind { Inherit, Fixed };
  Kind kind = Kind::Inherit;
  std::optional<core::Layout> layout;

  static ResultLayout inherit() {
    return {};
  }
  static ResultLayout fixed(core::Layout l) {
    return {Kind::Fixed, std::move(l)};
  }
};

class Operator {
 public:
  virtual runtime::Error setup(
      runtime::Span<const graph::ConstantArg> constant_args) {
    return runtime::Error::Ok;
  }
  virtual runtime::Error prepare(
      runtime::Span<core::Tensor*> inputs,
      runtime::Span<core::Tensor*> outputs) {
    return runtime::Error::Ok;
  }
  virtual runtime::Error reshape(
      runtime::Span<const graph::TensorSpec> input_specs) {
    return runtime::Error::Ok;
  }
  virtual runtime::Error execute(
      runtime::Span<core::Tensor*> inputs,
      runtime::Span<core::Tensor*> outputs) = 0;

  // Layout contract, queried by the layout-propagation pass. `input_specs` are
  // the specs of the (non-null) inputs, in argument order.
  //
  // Defaults make an operator layout-agnostic: it accepts any input layout and
  // produces results whose layout is inherited (i.e. the default contiguous
  // layout unless a consumer requires otherwise). Returning fewer entries than
  // there are inputs/results means the remainder take the default.
  virtual std::vector<LayoutConstraint> required_input_layouts(
      runtime::Span<const graph::TensorSpec> input_specs) const {
    return {};
  }
  virtual std::vector<ResultLayout> result_layouts(
      runtime::Span<const graph::TensorSpec> input_specs) const {
    return {};
  }

  // Reports the layouts the propagation pass resolved for this op's inputs and
  // results (aligned to argument / result order; nullopt == default
  // contiguous). Flexible producers use this to learn the layout they must
  // emit.
  virtual void configure_layouts(
      runtime::Span<const std::optional<core::Layout>> resolved_inputs,
      runtime::Span<const std::optional<core::Layout>> resolved_outputs) {}

  virtual ~Operator() = default;
};

/*
 * Construct the in-tree operator for a call node. `input_specs` are the specs
 * of the node's non-null inputs in argument order. Returns nullptr if there is
 * no in-tree kernel for the node.
 */
std::unique_ptr<Operator> create_operator(
    const graph::CallOperatorNode& node,
    runtime::Span<const graph::TensorSpec> input_specs);

} // namespace executorch::backends::xnnpack::operators
