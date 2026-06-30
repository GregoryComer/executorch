#pragma once

#include <executorch/backends/xnnpack/runtime/cache/packed_weight_cache.h>

#include <xnnpack.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace executorch::backends::xnnpack::cache {

/*
 * Exposes the ABI-neutral PackedWeightCache through XNNPACK's
 * `xnn_weights_cache_provider` vtable, so the XNNPACK delegate can share the
 * very same packed-weight store as the in-tree kernels. This is the
 * XNNPACK-specific *shell*; all storage/dedup/refcounting lives in the core.
 *
 * Key translation -- XNNPACK addresses sources by the pointer it was handed at
 * tensor-define time, plus a per-ukernel `seed` and a packing `fingerprint_id`:
 *   source_id   = name registered for the kernel pointer (+ bias name)
 *   fingerprint = (seed, fingerprint_id) folded together
 * XNNPACK addresses packed buffers by an opaque offset; here that offset is an
 * index into `offsets_`, resolved back to bytes via the core on demand.
 */
class XnnWeightsCacheAdapter {
 public:
  explicit XnnWeightsCacheAdapter(PackedWeightCache& cache) : cache_(cache) {
    provider_.context = this;
    provider_.look_up = look_up;
    provider_.reserve_space = reserve_space;
    provider_.look_up_or_insert = look_up_or_insert;
    provider_.is_finalized = is_finalized;
    provider_.offset_to_addr = offset_to_addr;
    provider_.delete_cache = delete_cache;
  }

  // Register the name of an unpacked source buffer. Mirrors the delegate's
  // load_unpacked_data: XNNPACK echoes this pointer back inside the cache key,
  // and we map it to the source's stable name here.
  void register_source(const void* unpacked_ptr, std::string name) {
    ptr_to_name_[unpacked_ptr] = std::move(name);
  }

  xnn_weights_cache_t get() {
    return &provider_;
  }

 private:
  static size_t look_up(void* ctx, const xnn_weights_cache_look_up_key* k);
  static void* reserve_space(void* ctx, size_t n);
  static size_t look_up_or_insert(
      void* ctx,
      const xnn_weights_cache_look_up_key* k,
      void* ptr,
      size_t size);
  static bool is_finalized(void* ctx);
  static void* offset_to_addr(void* ctx, size_t offset);
  static enum xnn_status delete_cache(void* ctx);

  PackedWeightKey key_from(const xnn_weights_cache_look_up_key* k) const;
  size_t offset_of(const PackedWeightKey& key) const;

  PackedWeightCache& cache_;
  xnn_weights_cache_provider provider_{};
  std::unordered_map<const void*, std::string> ptr_to_name_;
  std::vector<PackedWeightKey> offsets_;
};

} // namespace executorch::backends::xnnpack::cache
