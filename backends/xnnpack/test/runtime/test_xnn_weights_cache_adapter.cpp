#include <executorch/backends/xnnpack/runtime/cache/xnn_weights_cache_adapter.h>

#include <executorch/backends/xnnpack/runtime/cache/packed_weight_cache.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using executorch::backends::xnnpack::cache::PackedWeightCache;
using executorch::backends::xnnpack::cache::XnnWeightsCacheAdapter;

namespace {

// Pack `bytes` through the provider vtable under `key`, returning the offset.
size_t pack_through_provider(
    xnn_weights_cache_provider* p,
    const xnn_weights_cache_look_up_key* key,
    const std::vector<uint8_t>& bytes) {
  void* buf = p->reserve_space(p->context, bytes.size());
  EXPECT_NE(buf, nullptr);
  std::memcpy(buf, bytes.data(), bytes.size());
  return p->look_up_or_insert(p->context, key, buf, bytes.size());
}

} // namespace

// Drives the shared PackedWeightCache through XNNPACK's provider vtable, the
// way xnn_create_runtime would.
TEST(TestXnnWeightsCacheAdapter, DrivesCoreThroughProviderVtable) {
  PackedWeightCache cache;
  XnnWeightsCacheAdapter adapter(cache);
  auto* p = adapter.get();

  // A fake unpacked kernel buffer; register its name as the delegate would.
  uint8_t kernel_w0[4] = {0, 0, 0, 0};
  adapter.register_source(kernel_w0, "w0");

  xnn_weights_cache_look_up_key key{};
  key.seed = 1;
  key.kernel = kernel_w0;
  key.bias = nullptr;
  key.fingerprint_id = 7;

  EXPECT_EQ(p->look_up(p->context, &key), SIZE_MAX); // miss
  size_t off = pack_through_provider(p, &key, {1, 2, 3, 4});
  EXPECT_EQ(off, 0u);

  // Hit: same key resolves to the same offset, and resolves to the bytes.
  EXPECT_EQ(p->look_up(p->context, &key), 0u);
  auto* addr = static_cast<const uint8_t*>(p->offset_to_addr(p->context, off));
  ASSERT_NE(addr, nullptr);
  EXPECT_EQ(addr[0], 1);
  EXPECT_EQ(cache.num_entries(), 1u);
}

TEST(TestXnnWeightsCacheAdapter, FingerprintIdKeysDistinctPackings) {
  PackedWeightCache cache;
  XnnWeightsCacheAdapter adapter(cache);
  auto* p = adapter.get();

  uint8_t kernel_w0[4] = {0, 0, 0, 0};
  adapter.register_source(kernel_w0, "w0");

  // Same source pointer, different packing fingerprint -> distinct entries.
  xnn_weights_cache_look_up_key a{};
  a.kernel = kernel_w0;
  a.fingerprint_id = 7;
  xnn_weights_cache_look_up_key b = a;
  b.fingerprint_id = 9;

  size_t off_a = pack_through_provider(p, &a, {1, 1});
  size_t off_b = pack_through_provider(p, &b, {2, 2});
  EXPECT_NE(off_a, off_b);
  EXPECT_EQ(cache.num_entries(), 2u);

  // A repeat pack of `a` dedups back to its original offset.
  size_t off_a2 = pack_through_provider(p, &a, {1, 1});
  EXPECT_EQ(off_a2, off_a);
  EXPECT_EQ(cache.num_entries(), 2u);
}
