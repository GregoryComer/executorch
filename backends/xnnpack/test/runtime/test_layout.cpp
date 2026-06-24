#include <gtest/gtest.h>

#include <executorch/backends/xnnpack/runtime/core/dtype.h>
#include <executorch/backends/xnnpack/runtime/core/layout.h>
#include <executorch/backends/xnnpack/runtime/core/tensor.h>

using namespace executorch::backends::xnnpack::core;
using executorch::runtime::Span;

namespace {

size_t bytes(Span<const uint64_t> sizes, DType dtype) {
  auto r = compute_storage_size(sizes, dtype);
  EXPECT_TRUE(r.ok());
  return r.get();
}

size_t bytes(
    Span<const uint64_t> sizes,
    DType dtype,
    const std::optional<Layout>& layout) {
  auto r = compute_storage_size(sizes, dtype, layout);
  EXPECT_TRUE(r.ok());
  return r.get();
}

// A trivial opaque scheme used to exercise the size_fn hook: the packed size is
// the contiguous element count rounded up to a multiple of params[0].
size_t round_up_size_fn(
    const OpaquePacked& self,
    Span<const uint64_t> sizes,
    DType /*dtype*/) {
  size_t n = 1;
  for (auto s : sizes) {
    n *= static_cast<size_t>(s);
  }
  uint32_t mult = self.params[0];
  if (mult == 0) {
    return n;
  }
  return ((n + mult - 1) / mult) * mult;
}

} // namespace

// --- Default (no layout) path is byte-identical to the legacy two-arg call ---

TEST(TestLayout, default_layout_matches_contiguous) {
  const uint64_t s[] = {2, 3, 4};
  Span<const uint64_t> sizes(s, 3);

  EXPECT_EQ(
      bytes(sizes, DType::Float32, std::nullopt), bytes(sizes, DType::Float32));
  EXPECT_EQ(
      bytes(sizes, DType::QInt8, std::nullopt), bytes(sizes, DType::QInt8));
  EXPECT_EQ(
      bytes(sizes, DType::QInt4, std::nullopt), bytes(sizes, DType::QInt4));
  EXPECT_EQ(
      bytes(sizes, DType::Int64, std::nullopt), bytes(sizes, DType::Int64));
}

// --- DimOrder does not change the byte count ---

TEST(TestLayout, dim_order_preserves_size) {
  const uint64_t s[] = {2, 3, 4, 5};
  Span<const uint64_t> sizes(s, 4);

  Layout l = DimOrder{{0, 2, 3, 1}}; // channels-last permutation
  EXPECT_EQ(bytes(sizes, DType::Float32, l), bytes(sizes, DType::Float32));
  EXPECT_EQ(bytes(sizes, DType::Float16, l), bytes(sizes, DType::Float16));
}

// --- OpaquePacked routes sizing through size_fn ---

TEST(TestLayout, opaque_packed_uses_size_fn) {
  const uint64_t s[] = {10};
  Span<const uint64_t> sizes(s, 1);

  OpaquePacked op;
  op.scheme_id = 1;
  op.params[0] = 16; // round up to multiple of 16
  op.size_fn = round_up_size_fn;
  Layout l = op;

  // 10 elements rounded up to 16.
  EXPECT_EQ(bytes(sizes, DType::QInt8, l), 16u);
}

TEST(TestLayout, opaque_packed_missing_size_fn_errors) {
  const uint64_t s[] = {10};
  Span<const uint64_t> sizes(s, 1);

  OpaquePacked op; // size_fn left null
  Layout l = op;
  auto r = compute_storage_size(sizes, DType::QInt8, l);
  EXPECT_FALSE(r.ok());
}

// --- Blocked is defined but unwired; sizing is not yet supported ---

TEST(TestLayout, blocked_sizing_not_supported) {
  const uint64_t s[] = {8, 8};
  Span<const uint64_t> sizes(s, 2);

  Layout l = Blocked{{8, 8}};
  auto r = compute_storage_size(sizes, DType::Float32, l);
  EXPECT_FALSE(r.ok());
}

// --- Equality (used by TensorSpec::operator==) ---

TEST(TestLayout, equality) {
  EXPECT_EQ((Layout{DimOrder{{0, 1, 2}}}), (Layout{DimOrder{{0, 1, 2}}}));
  EXPECT_FALSE((Layout{DimOrder{{0, 1, 2}}} == Layout{DimOrder{{0, 2, 1}}}));

  OpaquePacked a;
  a.scheme_id = 3;
  a.params[0] = 8;
  a.size_fn = round_up_size_fn;
  OpaquePacked b = a;
  EXPECT_EQ(Layout{a}, Layout{b});

  b.params[0] = 9;
  EXPECT_FALSE(Layout{a} == Layout{b});

  // Different variant alternatives are never equal.
  EXPECT_FALSE(Layout{DimOrder{{0}}} == Layout{Blocked{{0}}});
}
