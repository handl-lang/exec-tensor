# How-to guides

Task recipes. Each assumes sibling checkouts of this repo and of the
platform repos it patches (`host` for `handl-platform` /
`handl-plugin-lib-handl`, `kernel` for `handl-value`).

## Compose tensors into a host

Build a `HandlModule` over `TENSOR_HNL` and attach the registry, then
register the module with the runtime's `ModuleRegistry` — after that,
`.hnl` code calls `tensor.*` like any other host function:

```rust
let module = HandlModule::new("tensor", TENSOR_HNL, &[])
    .with_ops(TENSOR_OPS, Arc::new(tensor_op_registry()));
```

The `.hnl` side declares the dep in `.handl.toml`
(`[dependencies] host-tensor = { path = "../host-tensor/src" }`) and
imports it (`from "host-tensor" import * as tensor`). See
[the reference](/host-tensor/reference/) for the full wiring.

## Call tensors from Python

With `--features python` the crate builds a PyO3 cdylib exposing
`ops()` (the declared surface) and `call(op_name, args_json)` (one op
dispatch over the same registry) — Python consumers load it as the
`handl_host_tensor` extension module.

## Add or change a tensor op

1. Declare it in `src/index.hnl` first — the declaration is the
   contract the runtime loads.
2. Implement the Rust `Function` in `src/ops/` (the handwritten core
   files for core ops; `aten_generated.rs`/`aten_kernels.rs` for the
   ATen catalog) and register it in `tensor_op_registry()`.
3. Keep the wire codec honest: tensors are `{"__tensor__": …}` objects
   and non-finite `f64`s spell as `{"__nonfinite__": …}` — see
   [the reference](/host-tensor/reference/#wire-codec).
4. `cargo test` — the drift-guard test fails if `index.hnl` and the
   registry disagree.

## Get gradients

Adjoints are registered pairs, not autodiff: `tensor.derive(name)`
returns the qualified name of the adjoint op for a primal (or `null`),
and the `*_backward` ops take the primal's args plus the upstream
gradient and return grads keyed by parameter name. Eleven pairs are
registered today (`add`/`mul`/`sub`/`div`, `matmul`, `conv2d`,
`max_pool`, `reshape`, `transpose`, `concat`, `batch_norm`).

## Handle a seam op

Some ATen-catalog ops (`rand`, `normal`, `empty`, …) are declared seams
that return an honest error naming the unported `aten::*` signature.
To land one, write the handwritten kernel in `aten_kernels.rs`, wire it
in `aten_generated.rs`, and remove the seam stub — the smoke test pins
the loud failure until then.
