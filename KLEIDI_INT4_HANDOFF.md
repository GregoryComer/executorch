# KleidiAI int4 in-tree kernels — session handoff

Status: **implemented, compiles on x86 (and against real kai headers); numeric
validation pending on Arm.** This doc is for picking the work up on an Arm
machine. Author: prototype built with Claude.

Branch: `kleidi-in-tree-layout`, 9 commits on top of the XNNPACK graph-runtime
stack (base `75b19350a1`). Nothing has been submitted or pushed.

Companion design doc (more detail on the layout framework):
`backends/xnnpack/runtime/KLEIDI_LAYOUT_PLAN.md`.

---

## Goal

Run dynamically-quantized int4 linears (dynamic int8 activations × blockwise
int4 weights, `qsi4c32p`) on KleidiAI kernels via the new **in-tree operator
API** of the XNNPACK/CPU graph runtime (`backends/xnnpack/runtime/`), instead of
delegating them to XNNPACK.

KleidiAI needs custom **RHS (weight) packing** and **LHS (activation) packing**.
The LHS pack is fused with dynamic int8 quantization and is done by an explicit
**dynamic-quant op**; the int4 GEMM is a second op. Both run in-tree, with the
packed LHS flowing between them.

This was built as the first consumer of a **general tensor-layout framework**, so
later kernels (channels-last convs, other packed GEMMs) reuse the same passes.

---

## Architecture

### Layout framework (general, reusable)

- `core::Layout` (`runtime/core/layout.h`) — a tagged union describing how a
  value's logical elements are physically arranged, orthogonal to the quant
  scheme (which stays in `QuantParams`). `nullopt` = default row-major contiguous
  (unchanged behavior). Tiers: `DimOrder` (NCHW/NHWC — defined, unwired),
  `Blocked` (defined, unwired), `OpaquePacked` (implemented; what Kleidi uses).
  `OpaquePacked` carries `scheme_id`, geometry `params[]`, and a plain
  function-pointer `size_fn` so `core` has no dependency on the packing module
  and packed buffers can be sized for dynamic shapes.
- `compute_storage_size(sizes, dtype, layout)` consults the layout; threaded
  through `TensorSpec`, `core::Tensor`, memory planning, and the executor.
- `Operator` layout contract (`runtime/operators/operator.h`):
  `required_input_layouts` (Any / Contiguous / Fixed per input),
  `result_layouts` (Inherit / Fixed), `configure_layouts` (reports resolved
  layouts back so a *flexible producer* learns what to emit). Defaults =
  layout-agnostic.

### Compile pipeline (additions in **bold**)

```
create_execution_plan:
  partition_xnn_subgraphs        # routing: keep int4 pattern out of XNN subgraph
  -> **instantiate_operators**   # build in-tree ops up front (runtime/plan/layout_assignment)
  -> **propagate_layouts**       # push Fixed input layouts onto producers; configure flexible producers
  -> schedule -> assign_value_slots
  -> create_plan_steps           # moves the pre-built op into each RunOperatorStep
Executor::build -> op->prepare() (RHS pack)   # then run: dq op packs LHS -> GEMM matmul
```

### Routing (`runtime/plan/xnn_support.cpp`, `prefer_in_tree_kernel`)

Routes a node in-tree (out of the XNNPACK subgraph) when it is:
- a `Linear` with a blockwise int4 weight (`QInt4` + `PerBlock`) and a
  dynamically-quantized int8 activation, for which `select_qsi4c32p_ukernel`
  succeeds; or
- the `Quantize` (convert → qdint8) feeding such a linear — **only if all its
  users are in-tree int4 linears** (never hand a Kleidi-packed buffer to an op
  that can't read it).

On x86 / no-Kleidi / unsupported CPU, selection fails → the pattern falls back
to XNNPACK delegation unchanged.

### The two ops

- `LinearInt4` (`runtime/operators/kleidi/linear_int4.{h,cpp}`): derives + caches
  its `KaiUkernelConfig` from its specs + CPU (so **nothing kleidi-specific is
  stored in the graph IR**); `required_input_layouts` = `[Fixed(lhs), Any, …]`;
  `prepare()` packs the weight into an **internal** buffer via
  `kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0`; `execute()` runs the matmul.
- `DynamicQuantPack` (`runtime/operators/kleidi/dynamic_quant_pack.{h,cpp}`): a
  flexible producer; `configure_layouts` hands it the LHS `OpaquePacked` layout
  its consumer requires; `execute()` runs `kai_run_lhs_quant_pack_qai8dxp_f32`.

### Ukernel layer (`runtime/operators/kleidi/kai_ukernel.{h,cpp}`)

kai-free public header; concrete kai variants + interface structs live in the
`.cpp` keyed by `variant_id`. Registered variants: i8mm and dotprod (SME2 not
yet registered). Compiles to abort stubs when KleidiAI is not built in
(`kai_available.{h,cpp}` exposes `kleidi_compiled_in()`).

### Build gating

`ENABLE_XNNPACK_KLEIDI` is defined only when the `kleidiai` CMake target exists
(ARM64), and `xnnpack_backend` links it (CMake). Buck: `-c
executorch.xnnpack_kleidi=1`. Source list of record:
`shim_et/xplat/executorch/build/build_variables.bzl` (`XNNPACK_BACKEND_BUCK_SRCS`).

---

## Commit stack (base `75b19350a1`)

| commit | summary |
|--------|---------|
| `4a551f61f3` | KleidiAI build plumbing |
| `feb366a270` | tensor layout primitive (`core::Layout`) |
| `56910280f8` | KleidiAI int4 ukernel layer |
| `2f88872a67` | operator layout contract + `create_operator(node, specs)` |
| `35b40b7062` | route int4 dynamic linear in-tree |
| `563027f32d` | operator instantiation + layout propagation passes |
| `c9e614233a` | int4 GEMM operator |
| `5cb7437453` | dynamic-quant LHS-pack operator |
| `a254d382d1` | blockwise int4 dynamic linear e2e test |

---

## Build & run on Arm

```bash
cd ~/src/executorch
cmake -S . -B cmake-out-arm \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_XNNPACK_ENABLE_KLEIDI=ON \
  -DEXECUTORCH_BUILD_TESTS=ON
cmake --build cmake-out-arm -j \
  --target backends_xnnpack_graph_e2e_test backends_xnnpack_graph_runtime_test

# Parity test for the in-tree int4 path:
cmake-out-arm/backends/xnnpack/test/backends_xnnpack_graph_e2e_test \
  --gtest_filter='TestE2E.linear_qd8_qsi4c32p_blockwise_dynamic'

# Unit tests for the new pieces:
cmake-out-arm/backends/xnnpack/test/backends_xnnpack_graph_runtime_test \
  --gtest_filter='TestLayout*:TestKaiUkernel*:TestLinearInt4*:TestRouting*'
```

- **KleidiAI source:** XNNPACK auto-downloads its pinned commit at configure
  time. If GitHub is blocked, copy a checkout over and add
  `-DKLEIDIAI_SOURCE_DIR=/path/to/kleidiai` (a local checkout exists at
  `~/src/kleidiai` @ `b431044a4407` on the dev box; the kai API used is present
  in both that and XNNPACK's pin — prefer XNNPACK's pin).
- **Confirm the in-tree path is taken:** the executor logs one line per step.
  Expect **two `OpStep[...]`** lines (dynamic-quant pack + int4 GEMM) for the
  blockwise test. A single `XnnStep` means it fell back to delegation (selection
  failed — e.g. CPU lacks dotprod/i8mm), which still passes numerically.

---

## Remaining work

1. **Numeric validation on Arm** — run the tests above. If the e2e test fails,
   the likely culprits are all in `linear_int4.cpp` / `kai_ukernel.cpp`:
   - int4 weight **nibble ordering** vs `kai_run_rhs_pack_nxk` (low nibble =
     k+0, value+8). Must match the exported `qsi4c32p` weight layout.
   - **bf16 scale** format and `scale_stride = (k/bl) * sizeof(uint16_t)`.
   - `m/n/k` derivation and `dst_stride_row = n * sizeof(float)`.
   - clamp (`output_min/max`) and pack params (`lhs_zero_point=1`,
     `rhs_zero_point=8`) — currently match the KleidiAI example.
2. **SME2 variant** not registered in `kai_ukernel.cpp` (only i8mm + dotprod).
3. **Per-channel int4** (`qsi4cxp`) not wired — only blockwise (`qsi4c32p`).
4. **Layout conflict policy** is "error" (v1). Relayout-op insertion for a value
   feeding consumers with different layouts is not implemented.
5. **Tiers 1/2** (`DimOrder` for NCHW/NHWC convs, `Blocked`) are defined but
   unwired — the framework supports them; no kernel uses them yet.
6. **Buck** Kleidi wiring (`_kleidi_deps()` in `backends/xnnpack/targets.bzl`)
   references `third_party_dep("kleidiai")` — verify that target resolves in
   fbsource; OSS Buck has no kleidiai target.
7. **Not submitted / pushed.** Create the stack when validated.
8. Only `-fsyntax-only` checks were run on x86; a full `cmake … && ctest`
   (delegated path) on x86 is also worth running as a sanity check.

---

## Watch-outs for whoever continues

- Commits were made with a pre-commit `lintrunner` hook that reformats and
  aborts the first attempt; re-`git add` + commit (or `--amend --no-verify` to
  fix a message).
- `shim_et/xplat/executorch/build/` is gitignored but `build_variables.bzl` is
  tracked — edit it directly to register new sources.
- The graph-runtime tests are **CMake-only** (not in `backends/xnnpack/test/targets.bzl`).
