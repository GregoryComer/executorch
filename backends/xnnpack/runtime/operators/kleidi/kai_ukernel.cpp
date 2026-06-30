#include <executorch/backends/xnnpack/runtime/operators/kleidi/kai_ukernel.h>

#include <executorch/runtime/platform/assert.h>

#if defined(ENABLE_XNNPACK_KLEIDI)

#include <cpuinfo.h>

#include <kai/kai_common.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h>
#include <kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h>
#include <kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h>
#include <kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h>

#endif // ENABLE_XNNPACK_KLEIDI

namespace executorch::backends::xnnpack::operators::kleidi {

using core::DType;
using core::OpaquePacked;

// OpaquePacked::params layout for both schemes. Shared by the layout factories
// and lhs_config_from_layout so the encoding lives in one place.
enum ParamIdx { kMr = 0, kNr = 1, kKr = 2, kSr = 3, kBl = 4, kScaleDt = 5 };

KaiUkernelConfig lhs_config_from_layout(const OpaquePacked& layout) {
  KaiUkernelConfig cfg;
  // LHS packing only needs mr/kr/sr (and bl is carried for completeness); the
  // matmul variant id is not required to pack the activation.
  cfg.mr = layout.params[kMr];
  cfg.kr = layout.params[kKr];
  cfg.sr = layout.params[kSr];
  cfg.bl = layout.params[kBl];
  return cfg;
}

#if defined(ENABLE_XNNPACK_KLEIDI)

namespace {

// Scales are stored as bf16, matching the blockwise int4 convention used by the
// delegated path.
constexpr enum kai_datatype kScaleDataType = kai_dt_bf16;

// A registered ukernel variant: its interface struct plus the CPU feature it
// requires. The order of this table defines KaiUkernelConfig::variant_id.
struct Variant {
  kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel ukernel;
  bool (*supported)();
};

#define KAI_VARIANT(suffix)                                  \
  {                                                          \
    kai_get_m_step_matmul_clamp_f32_##suffix,                \
        kai_get_n_step_matmul_clamp_f32_##suffix,            \
        kai_get_mr_matmul_clamp_f32_##suffix,                \
        kai_get_nr_matmul_clamp_f32_##suffix,                \
        kai_get_kr_matmul_clamp_f32_##suffix,                \
        kai_get_sr_matmul_clamp_f32_##suffix,                \
        kai_get_lhs_packed_offset_matmul_clamp_f32_##suffix, \
        kai_get_rhs_packed_offset_matmul_clamp_f32_##suffix, \
        kai_get_dst_offset_matmul_clamp_f32_##suffix,        \
        kai_get_dst_size_matmul_clamp_f32_##suffix,          \
        kai_run_matmul_clamp_f32_##suffix                    \
  }

bool has_i8mm() {
  return cpuinfo_has_arm_i8mm();
}
bool has_dotprod() {
  return cpuinfo_has_arm_neon_dot();
}

// Preference order: wider i8mm GEMM first, dotprod fallback.
const Variant kVariants[] = {
    {KAI_VARIANT(qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm), has_i8mm},
    {KAI_VARIANT(qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod), has_dotprod},
};
constexpr size_t kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);

const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel& ukernel_for(
    const KaiUkernelConfig& cfg) {
  ET_CHECK(cfg.variant_id < kNumVariants);
  return kVariants[cfg.variant_id].ukernel;
}

size_t lhs_size_fn(
    const OpaquePacked& self,
    runtime::Span<const uint64_t> sizes,
    DType /*dtype*/) {
  size_t k =
      sizes.size() == 0 ? 0 : static_cast<size_t>(sizes[sizes.size() - 1]);
  size_t m = 1;
  for (size_t i = 0; i + 1 < sizes.size(); i++) {
    m *= static_cast<size_t>(sizes[i]);
  }
  return kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
      m, k, self.params[kMr], self.params[kKr], self.params[kSr]);
}

size_t rhs_size_fn(
    const OpaquePacked& self,
    runtime::Span<const uint64_t> sizes,
    DType /*dtype*/) {
  size_t n = sizes.size() > 0 ? static_cast<size_t>(sizes[0]) : 0;
  size_t k = sizes.size() > 1 ? static_cast<size_t>(sizes[1]) : 0;
  return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
      n,
      k,
      self.params[kNr],
      self.params[kKr],
      self.params[kSr],
      self.params[kBl],
      static_cast<enum kai_datatype>(self.params[kScaleDt]));
}

OpaquePacked make_layout(
    const KaiUkernelConfig& cfg,
    KaiScheme scheme,
    size_t (
        *size_fn)(const OpaquePacked&, runtime::Span<const uint64_t>, DType)) {
  OpaquePacked op;
  op.scheme_id = scheme;
  op.params[kMr] = cfg.mr;
  op.params[kNr] = cfg.nr;
  op.params[kKr] = cfg.kr;
  op.params[kSr] = cfg.sr;
  op.params[kBl] = cfg.bl;
  op.params[kScaleDt] = static_cast<uint32_t>(kScaleDataType);
  op.size_fn = size_fn;
  return op;
}

} // namespace

std::optional<KaiUkernelConfig>
select_qsi4c32p_ukernel(uint64_t n, uint64_t k, uint32_t block_size) {
  (void)n;
  cpuinfo_initialize();

  // KleidiAI qsi4c32p requires a block length that is a multiple of 32 and
  // evenly divides K.
  if (block_size == 0 || block_size % 32 != 0 || k == 0 ||
      k % block_size != 0) {
    return std::nullopt;
  }

  for (uint32_t v = 0; v < kNumVariants; v++) {
    if (!kVariants[v].supported()) {
      continue;
    }
    const auto& uk = kVariants[v].ukernel;
    KaiUkernelConfig cfg;
    cfg.variant_id = v;
    cfg.mr = static_cast<uint32_t>(uk.get_mr());
    cfg.nr = static_cast<uint32_t>(uk.get_nr());
    cfg.kr = static_cast<uint32_t>(uk.get_kr());
    cfg.sr = static_cast<uint32_t>(uk.get_sr());
    cfg.bl = block_size;
    return cfg;
  }
  return std::nullopt;
}

OpaquePacked lhs_layout(const KaiUkernelConfig& cfg) {
  return make_layout(cfg, kSchemeQai8dxpLhs, lhs_size_fn);
}

OpaquePacked rhs_layout(const KaiUkernelConfig& cfg) {
  return make_layout(cfg, kSchemeQsi4c32pRhs, rhs_size_fn);
}

size_t lhs_packed_size(const KaiUkernelConfig& cfg, size_t m, size_t k) {
  return kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
      m, k, cfg.mr, cfg.kr, cfg.sr);
}

void run_lhs_quant_pack(
    const KaiUkernelConfig& cfg,
    size_t m,
    size_t k,
    const float* lhs,
    size_t lhs_stride,
    void* lhs_packed) {
  kai_run_lhs_quant_pack_qai8dxp_f32(
      m,
      k,
      cfg.mr,
      cfg.kr,
      cfg.sr,
      /*m_idx_start=*/0,
      lhs,
      lhs_stride,
      lhs_packed);
}

uint64_t qsi4c32p_packing_fingerprint(const KaiUkernelConfig& cfg) {
  // FNV-1a-ish fold of: scheme marker, KleidiAI version, ukernel geometry, and
  // (on SME hardware) the runtime vector length the packed layout depends on.
  uint64_t fp = 1469598103934665603ull;
  auto fold = [&fp](uint64_t v) { fp = (fp ^ v) * 1099511628211ull; };
  fold(kSchemeQsi4c32pRhs);
  for (const char* v = kai_get_version(); v != nullptr && *v != '\0'; ++v) {
    fold(static_cast<uint64_t>(static_cast<unsigned char>(*v)));
  }
  fold(cfg.variant_id);
  fold(cfg.nr);
  fold(cfg.kr);
  fold(cfg.sr);
  fold(cfg.bl);
  // TODO(kleidi-sme): when SME variants are registered, also fold in the
  // streaming vector length (kai_get_sme_vector_length_*) -- the SME packed
  // layout depends on it, so a buffer packed at one VL must not be reused at
  // another. The accessor is arch-gated, so it can't be called unconditionally
  // from this generic translation unit.
  return fp;
}

size_t rhs_packed_size(const KaiUkernelConfig& cfg, size_t n, size_t k) {
  return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
      n, k, cfg.nr, cfg.kr, cfg.sr, cfg.bl, kScaleDataType);
}

void run_rhs_pack(
    const KaiUkernelConfig& cfg,
    size_t n,
    size_t k,
    const uint8_t* rhs,
    const float* bias,
    const void* scale,
    size_t scale_stride,
    void* rhs_packed) {
  const struct kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params = {
      /*lhs_zero_point=*/1,
      /*rhs_zero_point=*/8,
      /*scale_dt=*/kScaleDataType,
  };
  kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
      /*num_groups=*/1,
      n,
      k,
      cfg.nr,
      cfg.kr,
      cfg.sr,
      cfg.bl,
      rhs,
      /*rhs_stride=*/(k / 2),
      bias,
      scale,
      scale_stride,
      rhs_packed,
      /*extra_bytes=*/0,
      &params);
}

void run_matmul(
    const KaiUkernelConfig& cfg,
    size_t m,
    size_t n,
    size_t k,
    const void* lhs_packed,
    const void* rhs_packed,
    float* dst,
    size_t dst_stride_row,
    float clamp_min,
    float clamp_max) {
  ukernel_for(cfg).run_matmul(
      m,
      n,
      k,
      cfg.bl,
      lhs_packed,
      rhs_packed,
      dst,
      dst_stride_row,
      /*dst_stride_col=*/sizeof(float),
      clamp_min,
      clamp_max);
}

#else // !ENABLE_XNNPACK_KLEIDI

std::optional<KaiUkernelConfig>
select_qsi4c32p_ukernel(uint64_t, uint64_t, uint32_t) {
  return std::nullopt;
}

OpaquePacked lhs_layout(const KaiUkernelConfig&) {
  ET_CHECK_MSG(false, "KleidiAI is not compiled in");
  return {};
}

OpaquePacked rhs_layout(const KaiUkernelConfig&) {
  ET_CHECK_MSG(false, "KleidiAI is not compiled in");
  return {};
}

size_t lhs_packed_size(const KaiUkernelConfig&, size_t, size_t) {
  ET_CHECK_MSG(false, "KleidiAI is not compiled in");
  return 0;
}

void run_lhs_quant_pack(
    const KaiUkernelConfig&,
    size_t,
    size_t,
    const float*,
    size_t,
    void*) {
  ET_CHECK_MSG(false, "KleidiAI is not compiled in");
}

uint64_t qsi4c32p_packing_fingerprint(const KaiUkernelConfig&) {
  return 0;
}

size_t rhs_packed_size(const KaiUkernelConfig&, size_t, size_t) {
  ET_CHECK_MSG(false, "KleidiAI is not compiled in");
  return 0;
}

void run_rhs_pack(
    const KaiUkernelConfig&,
    size_t,
    size_t,
    const uint8_t*,
    const float*,
    const void*,
    size_t,
    void*) {
  ET_CHECK_MSG(false, "KleidiAI is not compiled in");
}

void run_matmul(
    const KaiUkernelConfig&,
    size_t,
    size_t,
    size_t,
    const void*,
    const void*,
    float*,
    size_t,
    float,
    float) {
  ET_CHECK_MSG(false, "KleidiAI is not compiled in");
}

#endif // ENABLE_XNNPACK_KLEIDI

} // namespace executorch::backends::xnnpack::operators::kleidi
