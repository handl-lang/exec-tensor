# Explanation

## Why tensors are host functions, not language ops

Tensor code is the first real numerical workload on the platform, and
it belongs *outside* the language. A tensor has no literal syntax and
no eval-vocabulary constructor — it cannot be expressed as a pure
handl arrow — so every `tensor.*` op is an **FFI atom**: declared with
`op`/`handle` in `src/index.hnl` and implemented as a Rust `Function`
behind the `call_host` boundary. A host composes this component only if
it needs it; the `.hnl` surface stays the same whether the backend is
this dense `f64` array or something faster later.

Declaring the surface in `.hnl` keeps documentation and
implementation from drifting apart: `src/index.hnl` is the contract,
`src/ops/` implements it, and a drift-guard test fails the build when
they disagree.

## Why the wire is a tagged object

On the `serde_json::Value` boundary a tensor is just data:
`{"__tensor__": {"shape": [...], "data": [...], "dtype": "f64"}}` —
the same tagged convention as `{"__bytes__": …}` (ADR-0083 §2.1). The
tag keeps a tensor unambiguous next to user records, and engines
diverge honestly: an engine carrying the kernel `TaggedValue::Tensor`
decodes the tag at `call_host` and carries the tensor opaquely; an
engine without it sees a plain object. The wire shape is identical
either way, so a tensor can cross engine boundaries without loss.

Non-finite `f64`s get the same treatment — `{"__nonfinite__": "nan" |
"inf" | "-inf"}` — because `serde_json` has no NaN/±inf literal.
Decode accepts both spellings, so `tensor.div` by zero produces
`inf`/`NaN` results that round-trip exactly instead of degrading to
`null`.

## Why an ATen catalog

The handwritten core (31 ops) covers what the platform's own
workloads need — construction, elementwise algebra, `matmul`,
`conv2d`/`max_pool`/`batch_norm`, shape ops, and their adjoints. The
generated ATen catalog (~200 more ops) ports the PyTorch `aten::*`
vocabulary so `.hnl` code — and the optimizer's lowered artifacts —
can spell the ops model exporters already emit. It is generated glue
over handwritten kernels in `aten_kernels.rs`, and where a port is not
honest yet (RNG — nondeterminism would break verification; uninitialized
`empty` — the wire needs concrete values) the op is a declared seam
that fails loudly rather than faking behavior.

## Derivatives are a table, not autodiff

`tensor.derive(name)` resolves a primal to its registered adjoint op
name — eleven hand-maintained pairs today. A true `derive` over `.hnl`
functions would need source-to-source AD or a reverse-mode tape in the
eval engine; both live outside this component and are deferred. The
pairs table is the honest minimal surface: an adjoint is just another
`Function` in the same registry.

## Where it sits

This crate is a **host-side component**, not an engine and not a
language feature: runtimes compose it into their module chain
(`HandlModule::with_ops`), and `.hnl` code reaches it through ordinary
host calls. See the [reference](/host-tensor/reference/) for the API
and the [how-to guides](/host-tensor/how-to/) for tasks.
