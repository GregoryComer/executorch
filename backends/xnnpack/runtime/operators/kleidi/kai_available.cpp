#include <executorch/backends/xnnpack/runtime/operators/kleidi/kai_available.h>

#if defined(ENABLE_XNNPACK_KLEIDI)
// Validates that the kleidiai library's include path is wired up. The concrete
// ukernel headers are pulled in by the kleidi ukernel layer (see kleidi/).
#include <kai/kai_common.h>
#endif

namespace executorch::backends::xnnpack::operators::kleidi {

bool kleidi_compiled_in() {
#if defined(ENABLE_XNNPACK_KLEIDI)
  return true;
#else
  return false;
#endif
}

} // namespace executorch::backends::xnnpack::operators::kleidi
