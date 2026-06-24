#pragma once

namespace executorch::backends::xnnpack::operators::kleidi {

/*
 * Returns true if KleidiAI in-tree kernels were compiled into this build, i.e.
 * the kleidiai library is linked and its kai/ headers are available. This is
 * the single gate the kernel-selection pass consults before choosing a Kleidi
 * ukernel; everything else in the kleidi/ module is only reachable when this is
 * true.
 */
bool kleidi_compiled_in();

} // namespace executorch::backends::xnnpack::operators::kleidi
