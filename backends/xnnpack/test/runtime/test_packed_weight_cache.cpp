#include <executorch/backends/xnnpack/runtime/cache/packed_weight_cache.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using executorch::backends::xnnpack::cache::PackedWeightCache;
using executorch::backends::xnnpack::cache::PackedWeightKey;
using executorch::runtime::Error;

namespace {

// Pack `bytes` into the cache under `key` via the reserve/commit contract.
const uint8_t* pack(
    PackedWeightCache& cache,
    const PackedWeightKey& key,
    const std::vector<uint8_t>& bytes) {
  auto reserved = cache.reserve(bytes.size());
  EXPECT_EQ(reserved.error(), Error::Ok);
  void* buf = reserved.get();
  std::memcpy(buf, bytes.data(), bytes.size());
  auto committed = cache.commit(key, buf, bytes.size());
  EXPECT_EQ(committed.error(), Error::Ok);
  return committed.get().data();
}

} // namespace

TEST(TestPackedWeightCache, ReserveCommitLookupRoundTrip) {
  PackedWeightCache cache;
  PackedWeightKey key{"w0", 0x1234};
  std::vector<uint8_t> bytes{1, 2, 3, 4, 5};

  EXPECT_FALSE(cache.lookup(key).has_value());
  const uint8_t* committed = pack(cache, key, bytes);

  auto hit = cache.lookup(key);
  ASSERT_TRUE(hit.has_value());
  EXPECT_EQ(hit->size(), bytes.size());
  EXPECT_EQ(hit->data(), committed); // same buffer, no copy
  EXPECT_EQ(0, std::memcmp(hit->data(), bytes.data(), bytes.size()));
  EXPECT_EQ(cache.num_entries(), 1u);
}

TEST(TestPackedWeightCache, SameKeyDedups) {
  PackedWeightCache cache;
  PackedWeightKey key{"w0", 0x1234};

  const uint8_t* first = pack(cache, key, {1, 2, 3, 4});
  // A second pack of the same key drops the new buffer and returns the first.
  const uint8_t* second = pack(cache, key, {9, 9, 9, 9});

  EXPECT_EQ(cache.num_entries(), 1u);
  EXPECT_EQ(first, second);
  EXPECT_EQ(first[0], 1); // original bytes retained
}

TEST(TestPackedWeightCache, FingerprintDistinguishesPackings) {
  PackedWeightCache cache;
  // Same source, different packing fingerprint -> two independent entries.
  pack(cache, PackedWeightKey{"w0", 0x1111}, {1, 2});
  pack(cache, PackedWeightKey{"w0", 0x2222}, {3, 4});

  EXPECT_EQ(cache.num_entries(), 2u);
  auto a = cache.lookup(PackedWeightKey{"w0", 0x1111});
  auto b = cache.lookup(PackedWeightKey{"w0", 0x2222});
  ASSERT_TRUE(a.has_value() && b.has_value());
  EXPECT_EQ(a->data()[0], 1);
  EXPECT_EQ(b->data()[0], 3);
}

TEST(TestPackedWeightCache, FinalizeRefcountsAndReleaseFrees) {
  PackedWeightCache cache;
  PackedWeightKey key{"w0", 0x1234};
  pack(cache, key, {1, 2, 3});

  auto used = cache.finalize();
  ASSERT_EQ(used.size(), 1u);
  EXPECT_EQ(used[0], key);

  // Refcount is 1 after one build; releasing it frees the entry.
  cache.release({used.data(), used.size()});
  EXPECT_EQ(cache.num_entries(), 0u);
  EXPECT_FALSE(cache.lookup(key).has_value());
}

TEST(TestPackedWeightCache, RefcountSurvivesUntilAllBuildsRelease) {
  PackedWeightCache cache;
  PackedWeightKey key{"w0", 0x1234};

  // Two builds both use the same weight.
  pack(cache, key, {1, 2, 3});
  auto used_a = cache.finalize();
  pack(cache, key, {1, 2, 3}); // dedup hit, but counts as used by build B
  auto used_b = cache.finalize();

  cache.release({used_a.data(), used_a.size()});
  EXPECT_EQ(cache.num_entries(), 1u); // still held by build B
  cache.release({used_b.data(), used_b.size()});
  EXPECT_EQ(cache.num_entries(), 0u);
}
