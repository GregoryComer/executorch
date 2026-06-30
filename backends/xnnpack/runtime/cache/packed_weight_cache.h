#pragma once

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace executorch::backends::xnnpack::cache {

/*
 * Identity of a packed-weight buffer. Two parts, mirroring where upstream
 * XNNPACK's weights-cache key is heading (`xnn_weights_cache_look_up_key` now
 * carries a `seed` + `fingerprint_id`):
 *
 *  - source_id:   identity of the *unpacked* source. The PTD `named_key` in
 *                 production; for inline constants a content hash. (This v0
 *                 prototype uses the constant's data pointer as a stand-in.)
 *  - fingerprint: identity of the *packing* -- scheme + ukernel variant +
 *                 geometry (+ library version / SME vector length in
 *                 production). Distinguishes two packings of the same source so
 *                 a buffer packed under one config is never mistaken for one
 *                 packed under another (the correctness guarantee a name-only
 *                 key lacks). Analogous to XNNPACK's xnn_fingerprint.
 */
struct PackedWeightKey {
  std::string source_id;
  uint64_t fingerprint = 0;

  bool operator==(const PackedWeightKey& other) const {
    return fingerprint == other.fingerprint && source_id == other.source_id;
  }
};

/*
 * Backend-agnostic packed-weight cache: no xnn_* / kai_* types, so the in-tree
 * Kleidi kernels use it directly and the XNNPACK delegate can wrap it behind
 * the `xnn_weights_cache_provider` vtable. v0 prototype: in-memory heap store
 * with a reserve/commit/lookup contract (pack straight into cache memory, no
 * temp copy) and per-build reference counting for cross-method sharing.
 */
class PackedWeightCache {
 public:
  // Borrowed view of the packed bytes for `key` if present (marks it used by
  // the current build); std::nullopt on miss.
  std::optional<runtime::Span<const uint8_t>> lookup(const PackedWeightKey& key);

  // Reserve an uninitialized buffer of `bytes` to pack into. The returned
  // pointer is owned by the cache and stays valid until commit().
  runtime::Result<void*> reserve(size_t bytes);

  // Commit the buffer previously returned by reserve() under `key`. If `key` is
  // already cached the pending buffer is dropped and the existing bytes are
  // returned (dedup). Returns a borrowed view of the canonical packed bytes.
  runtime::Result<runtime::Span<const uint8_t>>
  commit(const PackedWeightKey& key, void* ptr, size_t bytes);

  // Marks the current build complete: bumps the refcount of every entry this
  // build touched and returns those keys so the owner can release() them at
  // teardown.
  std::vector<PackedWeightKey> finalize();

  // Decrements the refcount of each key; frees the entry when it reaches zero.
  void release(runtime::Span<const PackedWeightKey> keys);

  size_t num_entries() const {
    return entries_.size();
  }

 private:
  struct KeyHash {
    size_t operator()(const PackedWeightKey& k) const {
      return std::hash<std::string>()(k.source_id) ^
          (std::hash<uint64_t>()(k.fingerprint) * 1099511628211ull);
    }
  };

  struct Entry {
    std::vector<uint8_t> bytes;
    uint32_t ref_count = 0;
    bool in_current_build = false;
  };

  std::unordered_map<PackedWeightKey, Entry, KeyHash> entries_;
  // Buffers handed out by reserve() but not yet committed. Moving a std::vector
  // preserves data(), so neither growth of `pending_` nor moving the cache
  // invalidates an outstanding reserved pointer or a borrowed packed view.
  std::vector<std::vector<uint8_t>> pending_;
};

} // namespace executorch::backends::xnnpack::cache
