#include <executorch/backends/xnnpack/runtime/cache/xnn_weights_cache_adapter.h>

#include <cstdint>

namespace executorch::backends::xnnpack::cache {

PackedWeightKey XnnWeightsCacheAdapter::key_from(
    const xnn_weights_cache_look_up_key* k) const {
  std::string source_id;
  auto kernel = ptr_to_name_.find(k->kernel);
  if (kernel != ptr_to_name_.end()) {
    source_id = kernel->second;
  } else {
    source_id = "ptr:" +
        std::to_string(static_cast<unsigned long long>(
            reinterpret_cast<uintptr_t>(k->kernel)));
  }
  if (k->bias != nullptr) {
    auto bias = ptr_to_name_.find(k->bias);
    if (bias != ptr_to_name_.end()) {
      source_id += "+" + bias->second;
    }
  }
  // Fold the per-ukernel seed and the packing fingerprint into one value.
  uint64_t fingerprint = (static_cast<uint64_t>(k->seed) << 32) ^
      static_cast<uint64_t>(k->fingerprint_id);
  return PackedWeightKey{std::move(source_id), fingerprint};
}

size_t XnnWeightsCacheAdapter::offset_of(const PackedWeightKey& key) const {
  for (size_t i = 0; i < offsets_.size(); i++) {
    if (offsets_[i] == key) {
      return i;
    }
  }
  return SIZE_MAX;
}

size_t XnnWeightsCacheAdapter::look_up(
    void* ctx,
    const xnn_weights_cache_look_up_key* k) {
  auto* self = static_cast<XnnWeightsCacheAdapter*>(ctx);
  return self->offset_of(self->key_from(k));
}

void* XnnWeightsCacheAdapter::reserve_space(void* ctx, size_t n) {
  auto* self = static_cast<XnnWeightsCacheAdapter*>(ctx);
  auto reserved = self->cache_.reserve(n);
  return reserved.error() == runtime::Error::Ok ? reserved.get() : nullptr;
}

size_t XnnWeightsCacheAdapter::look_up_or_insert(
    void* ctx,
    const xnn_weights_cache_look_up_key* k,
    void* ptr,
    size_t size) {
  auto* self = static_cast<XnnWeightsCacheAdapter*>(ctx);
  PackedWeightKey key = self->key_from(k);
  // The core dedups: if the key already exists the freshly packed buffer is
  // dropped and the canonical bytes returned.
  auto committed = self->cache_.commit(key, ptr, size);
  if (committed.error() != runtime::Error::Ok) {
    return SIZE_MAX;
  }
  size_t offset = self->offset_of(key);
  if (offset == SIZE_MAX) {
    offset = self->offsets_.size();
    self->offsets_.push_back(std::move(key));
  }
  return offset;
}

bool XnnWeightsCacheAdapter::is_finalized(void* ctx) {
  // The core has no finalize lockout; inserts are always permitted (it is a
  // long-lived store shared across runtimes).
  (void)ctx;
  return false;
}

void* XnnWeightsCacheAdapter::offset_to_addr(void* ctx, size_t offset) {
  auto* self = static_cast<XnnWeightsCacheAdapter*>(ctx);
  if (offset >= self->offsets_.size()) {
    return nullptr;
  }
  auto bytes = self->cache_.lookup(self->offsets_[offset]);
  return bytes.has_value() ? const_cast<uint8_t*>(bytes->data()) : nullptr;
}

enum xnn_status XnnWeightsCacheAdapter::delete_cache(void* ctx) {
  // The core outlives this provider; nothing to free here.
  (void)ctx;
  return xnn_status_success;
}

} // namespace executorch::backends::xnnpack::cache
