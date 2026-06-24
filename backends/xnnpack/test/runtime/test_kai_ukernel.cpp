#include <gtest/gtest.h>

#include <executorch/backends/xnnpack/runtime/core/layout.h>
#include <executorch/backends/xnnpack/runtime/core/tensor.h>
#include <executorch/backends/xnnpack/runtime/operators/kleidi/kai_available.h>
#include <executorch/backends/xnnpack/runtime/operators/kleidi/kai_ukernel.h>

using namespace executorch::backends::xnnpack;
using namespace executorch::backends::xnnpack::operators::kleidi;
using executorch::runtime::Span;

// When KleidiAI is not compiled into this build, selection must always fail so
// the kernel-selection pass falls back to delegation.
TEST(KaiUkernel, selection_requires_compiled_in) {
  auto cfg = select_qsi4c32p_ukernel(/*n=*/64, /*k=*/256, /*block_size=*/32);
  if (!kleidi_compiled_in()) {
    EXPECT_FALSE(cfg.has_value());
  }
}

// Block-size constraints are enforced (only meaningful when compiled in; the
// host may also lack the required CPU features, in which case selection fails
// for that reason instead).
TEST(KaiUkernel, rejects_invalid_block_size) {
  if (!kleidi_compiled_in()) {
    GTEST_SKIP();
  }
  // Not a multiple of 32.
  EXPECT_FALSE(select_qsi4c32p_ukernel(64, 256, 30).has_value());
  // k not a multiple of the block size.
  EXPECT_FALSE(select_qsi4c32p_ukernel(64, 250, 32).has_value());
}

// The OpaquePacked layouts produced for a selected ukernel must size buffers
// identically to the ukernel layer's own packed-size helpers.
TEST(KaiUkernel, layout_sizes_match_wrappers) {
  if (!kleidi_compiled_in()) {
    GTEST_SKIP();
  }
  auto cfg = select_qsi4c32p_ukernel(/*n=*/64, /*k=*/256, /*block_size=*/32);
  if (!cfg.has_value()) {
    GTEST_SKIP(); // host lacks the required CPU features
  }

  // RHS: weight [n, k].
  const uint64_t rhs_sizes[] = {64, 256};
  auto rhs = core::Layout{rhs_layout(*cfg)};
  auto rhs_bytes = core::compute_storage_size(
      Span<const uint64_t>(rhs_sizes, 2), core::DType::QInt4, rhs);
  ASSERT_TRUE(rhs_bytes.ok());
  EXPECT_GT(rhs_bytes.get(), 0u);
  EXPECT_EQ(rhs_bytes.get(), rhs_packed_size(*cfg, 64, 256));

  // LHS: activation [m, k].
  const uint64_t lhs_sizes[] = {10, 256};
  auto lhs = core::Layout{lhs_layout(*cfg)};
  auto lhs_bytes = core::compute_storage_size(
      Span<const uint64_t>(lhs_sizes, 2), core::DType::QInt8, lhs);
  ASSERT_TRUE(lhs_bytes.ok());
  EXPECT_GT(lhs_bytes.get(), 0u);
  EXPECT_EQ(lhs_bytes.get(), lhs_packed_size(*cfg, 10, 256));
}
