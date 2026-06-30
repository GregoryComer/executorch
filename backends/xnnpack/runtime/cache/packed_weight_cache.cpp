#include <executorch/backends/xnnpack/runtime/cache/packed_weight_cache.h>

#include <utility>

namespace executorch::backends::xnnpack::cache {

using executorch::runtime::Error;
using executorch::runtime::Result;
using executorch::runtime::Span;

std::optional<Span<const uint8_t>> PackedWeightCache::lookup(
    const PackedWeightKey& key) {
  auto it = entries_.find(key);
  if (it == entries_.end()) {
    return std::nullopt;
  }
  it->second.in_current_build = true;
  return Span<const uint8_t>(it->second.bytes.data(), it->second.bytes.size());
}

Result<void*> PackedWeightCache::reserve(size_t bytes) {
  pending_.emplace_back(bytes);
  return static_cast<void*>(pending_.back().data());
}

Result<Span<const uint8_t>> PackedWeightCache::commit(
    const PackedWeightKey& key,
    void* ptr,
    size_t bytes) {
  size_t idx = SIZE_MAX;
  for (size_t i = 0; i < pending_.size(); i++) {
    if (pending_[i].data() == ptr) {
      idx = i;
      break;
    }
  }
  ET_CHECK_OR_RETURN_ERROR(
      idx != SIZE_MAX, InvalidArgument, "commit() pointer was not reserved");
  (void)bytes;

  // Dedup: an entry for this key already exists -- drop the freshly packed
  // buffer and hand back the canonical one.
  auto existing = entries_.find(key);
  if (existing != entries_.end()) {
    pending_.erase(pending_.begin() + idx);
    existing->second.in_current_build = true;
    return Span<const uint8_t>(
        existing->second.bytes.data(), existing->second.bytes.size());
  }

  Entry entry;
  entry.bytes = std::move(pending_[idx]);
  entry.in_current_build = true;
  pending_.erase(pending_.begin() + idx);

  auto inserted = entries_.emplace(key, std::move(entry)).first;
  return Span<const uint8_t>(
      inserted->second.bytes.data(), inserted->second.bytes.size());
}

std::vector<PackedWeightKey> PackedWeightCache::finalize() {
  std::vector<PackedWeightKey> used;
  for (auto& kv : entries_) {
    if (kv.second.in_current_build) {
      kv.second.ref_count++;
      kv.second.in_current_build = false;
      used.push_back(kv.first);
    }
  }
  pending_.clear();
  return used;
}

void PackedWeightCache::release(Span<const PackedWeightKey> keys) {
  for (const auto& key : keys) {
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      continue;
    }
    if (it->second.ref_count > 0 && --it->second.ref_count == 0) {
      entries_.erase(it);
    }
  }
}

} // namespace executorch::backends::xnnpack::cache
