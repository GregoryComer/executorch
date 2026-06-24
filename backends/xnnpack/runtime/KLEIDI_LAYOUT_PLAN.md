# KleidiAI int4 GEMM via in-tree kernels — design & commit plan

Status: prototype. This document describes how KleidiAI int4 GEMM kernels are
wired into the XNNPACK/CPU backend's in-tree operator API, and the general
tensor-layout framework that makes the int4 work the first of several packed /
re-laid-out kernel families (channels-last convs, blocked packing, etc.).

## Motivation & constraints

KleidiAI int4 GEMM (dynamically-quantized int8 activations × int4 weights)
requires:
- **RHS (weight) packing** into a ukernel-specific layout (once, at build).
- **LHS (activation) packing** that fuses dynamic int8 quantization with packing
  (per inference). We want this done by an *explicit dynamic-quant op*, also on
  Kleidi.
- The LHS pack, RHS pack, and matmul are all parameterized by the *same* ukernel
  variant's `(mr, nr, kr, sr)`. So the dynamic-quant op and the GEMM op must
  agree on one variant per linear.

Two frictions with the current in-tree pipeline:
1. `create_operator(graph::Operator)` only sees the op enum — not specs, quant
   params, or any selected config.
2. Memory planning sizes buffers as `shape × dtype`, but packed buffers are not
   that shape — so packing **layout must be a visible property of the value**.

## Layout taxonomy

`Layout` is an extensible tagged union on `TensorSpec` / `core::Tensor`. The
logical shape stays in `sizes`; `Layout` describes physical arrangement + inline
metadata. Quant *scheme* stays in `QuantParams` (orthogonal).

| Tier | Models | Size | Conversion | v1 |
|---|---|---|---|---|
| 0 default (`nullopt`) | row-major contiguous | shape×dtype | — | implemented |
| 1 `DimOrder` | axis permutation (minor-to-major) | shape×dtype | transpose | defined, unwired |
| 2 `Blocked` | tiled+padded geometry | from geometry | (un)pack | defined, unwired |
| 3 `Opaque` | kernel-private buffer {producer,consumer,size_fn} | size_fn | registered packer | implemented (Kleidi) |

Each kind provides two visitors: `size_bytes(shape, dtype)` and
`convert_from(src)` (the op, if any, that materializes it).

## Pipeline

```
partition_xnn_subgraphs
  -> select_kernels        : pick KaiUkernelConfig per in-tree node; in-tree iff selectable
  -> instantiate_operators : create_operator(node, specs, config); Operator owns its contract
  -> propagate_layouts     : assign Layout per value; resolve Inherit; insert relayout ops;
                             configure flexible producers
  -> schedule -> assign_value_slots -> create_memory_plan (layout-aware sizing)
  -> Executor::build : op->prepare() (RHS pack); run: dq op packs LHS -> GEMM matmul
```

- **Operator layout contract** (trait): `required_input_layouts()` (may be
  `Any`), `result_layouts()` (may be `Inherit`), `configure_layouts(resolved)`.
- **Flexible producer**: the dynamic-quant op declares `result = Inherit`;
  propagation pushes the consumer GEMM's required Kleidi LHS layout onto it.
- **Conflict policy (v1)**: constrain selection so shared-activation consumers
  agree; else fall the group back to XNNPACK delegation. Evolution: duplicate +
  insert converters (additive, since relayout is a real op).

## Commit stack

1. Build plumbing — populate KleidiAI submodule; expose `kai_*` behind
   `EXECUTORCH_XNNPACK_ENABLE_KLEIDIAI` (ARM64), CMake + Buck.
2. `Layout` primitive + layout-aware `compute_storage_size` (Tier-0/3 live).
3. Kleidi ukernel layer (`runtime/operators/kleidi/`): interface wrapper,
   cpuinfo variant selection, `KaiUkernelConfig`, packed-size helpers.
4. `Operator` layout-contract trait + earlier instantiation; `create_operator`
   refactor to receive node + specs + config.
5. `select_kernels` pass + routing (subsumes `prefer_in_tree_kernel`).
6. `propagate_layouts` pass.
7. Int4 GEMM op (RHS pack in `prepare()`, matmul in `execute()`, scalar fallback).
8. Dynamic-quant LHS-pack op (materialize explicit dq node if export doesn't —
   verify).
9. E2E parity tests vs delegated path.

Commits 2–6 are the reusable framework; 7–8 the int4 instantiation.
