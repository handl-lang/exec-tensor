# Reference

## Public API

The crate is `handl-host-tensor`. `.hnl` code reaches the ops through
the `tensor.*` names declared in `src/index.hnl`; Rust consumers use
the registry and module wiring below.

| item | what it is |
|---|---|
| `TENSOR_HNL` | the embedded `src/index.hnl` source — the declared surface |
| `TENSOR_OPS` | the op-name list the registry covers |
| `tensor_op_registry()` | the `OpRegistry` of Rust `Function` impls |
| `tensor.test.hnl` | the `.hnl` test corpus (`assert`-based) |

### Composing into a host

```rust
use std::sync::Arc;
use handl_plugin_lib_handl::HandlModule;
use handl_host_tensor::{TENSOR_HNL, TENSOR_OPS, tensor_op_registry};

let module = HandlModule::new("tensor", TENSOR_HNL, &[])
    .with_ops(TENSOR_OPS, Arc::new(tensor_op_registry()));
// register `module` in the runtime's ModuleRegistry / ModuleRuntime;
// `call_host("tensor.matmul")` now reaches the Rust impl.
```

A consumer repo declares the dep in `.handl.toml`:

```toml
[dependencies]
host-tensor = { path = "../host-tensor/src" }
```

```hnl
from "host-tensor" import * as tensor
tensor.matmul(tensor.from_list([[1, 2]]), tensor.from_list([[3], [4]]))
```

## The op catalog

`src/index.hnl` declares **239 `tensor.*` ops** — the source of truth;
the Rust impls in `src/ops/` mirror it (a drift-guard test pins the
interface). Two layers:

### Handwritten core (32 ops)

| group | ops |
|---|---|
| construction | `tensor.zeros`, `tensor.ones`, `tensor.from_list` |
| projection | `tensor.to_list`, `tensor.shape` |
| elementwise algebra | `tensor.add`/`mul`/`sub`/`div` — numpy broadcasting |
| adjoints | `tensor.add_backward`, `mul_backward`, `sub_backward`, `div_backward` |
| linalg | `tensor.matmul` (rank-2/3, batch broadcast), `tensor.matmul_backward`, `tensor.linear` |
| conv/norm | `tensor.conv2d`, `tensor.conv2d_backward`, `tensor.max_pool`, `tensor.max_pool_backward`, `tensor.batch_norm`, `tensor.batch_norm_backward` |
| shape | `tensor.reshape`, `reshape_backward`, `unsqueeze`, `select`, `transpose`, `transpose_backward`, `concat`, `concat_backward` |
| derivatives | `tensor.derive` — registered adjoint name for a primal, or `null` |
| ingest | `tensor.load(path[, name])` — one tensor out of a safetensors sidecar |

Broadcasting: trailing dims align, a `1`-dim stretches, a missing
leading dim counts as `1`; a mismatched non-`1` pair is an error.
Positional optionals take `null`/missing for defaults (`conv2d`'s
`bias?`/`stride?`/`padding?`/`groups?`, `batch_norm`'s `eps?`).

### ATen catalog (207 ops)

The PyTorch `aten::*` vocabulary, declared between the
`ATEN-PORT-GEN:BEGIN/END` markers in `src/index.hnl` and implemented in
`src/ops/aten_generated.rs` + `aten_kernels.rs`:

- **unary math** — `abs`, `neg`, `exp`/`exp2`/`expm1`, `log`/`log1p`/
  `log2`/`log10`, `sqrt`, `rsqrt`, `reciprocal`, `square`, the
  `sin`/`cos`/`tan`/`asin`/`acos`/`atan` family, `sinh`/`cosh`/`tanh`
  and inverses, `floor`/`ceil`/`round`/`trunc`/`frac`, `sign`,
  `erf`/`erfc`, `sinc`, `deg2rad`/`rad2deg`, …
- **activations** — `relu`, `leaky_relu`, `prelu`, `elu`, `celu`,
  `selu`, `gelu`, `silu`, `mish`, `softplus`, `softshrink`,
  `hardshrink`, `hardtanh`, `threshold`, `sigmoid`, `logit`, `glu`,
  `rms_norm`, `layer_norm`, …
- **elementwise binary** — `add`-family companions (`atan2`, `hypot`,
  `logaddexp`, `maximum`/`minimum`, `nextafter`, `fmod`,
  `floor_divide`, `copysign`, `pow`, `float_power`, …), comparisons
  (`eq`/`ne`/`lt`/`le`/`gt`/`ge`, `isfinite`/`isinf`/`isnan`/…),
  logicals (`logical_and`/`or`/`xor`/`not`), `where`, `clamp*`,
  `lerp`, `heaviside`, `nan_to_num`, …
- **reductions** — `sum`, `prod`, `mean`, `var`, `std`, `amax`, `amin`,
  `argmax`, `argmin`, `logsumexp`, `all`, `any`, `norm`, `median`,
  `quantile`, `nansum`, `nanmean`, `softmax`, `log_softmax`, …
- **indexing & shape** — `select`-family (`squeeze`, `unsqueeze`,
  `flatten`, `chunk`, `split`, `stack`, `unbind`, `repeat`, `flip`,
  `triu`/`tril`, `diag`/`diagonal`, `t`/`mT`, `swapaxes`, `moveaxis`,
  `rot90`, `pixel_shuffle`/`unshuffle`, padding), gathers
  (`index_select`, `gather`, `index_add`, `index_fill`, `take`,
  `nonzero`, `searchsorted`, `bincount`, `unique`,
  `topk`, `sort`, `argsort`, `msort`), …
- **linalg** — `mm`, `bmm`, `addmm`, `addbmm`, `baddbmm`, `mv`, `dot`,
  `vdot`, `ger`, `outer`, `addr`, `addmv`, `tensordot`, `kron`,
  `cross`, `einsum`, `inner`, `dist`, `cdist`, `pdist`, …
- **constructors & factories** — `full`, `full_like`, `zeros_like`,
  `ones_like`, `empty`, `rand`, `normal`, `arange`,
  `linspace`, `logspace`, `eye`, `meshgrid`, …
- **queries & misc** — `size`, `stride`, `numel`, `dim`, `item`,
  `is_contiguous`, `to`, `clone`, `detach`, `contiguous`, `view`,
  `allclose`, `is_nonzero`, `corrcoef`, `avg_pool2d`, …

**Declared seams.** A few catalog entries are honest stubs: they are
declared in `index.hnl` and registered, but the impl returns an error
naming the unported `aten::*` signature rather than faking behavior —
RNG (`rand`, `normal` — nondeterminism would break verification) and
uninitialized fill (`empty` — the wire needs concrete values). The
smoke test pins that they fail loudly.

See `src/index.hnl` for the authoritative per-op signatures — the
generated block mirrors the ATen schema 1:1, including `int[1]?`-style
optional-dim arguments.

## Wire codec

Tensors cross the `serde_json::Value` boundary tagged, in the
`{"__bytes__": …}` style (ADR-0083 §2.1):

```json
{"__tensor__": {"shape": [2, 3], "data": [1, 2, 3, 4, 5, 6], "dtype": "f64"}}
```

- `shape` — extent per axis (rank = `shape.len()`); may be empty for a
  scalar.
- `data` — flat row-major `f64` payload; `data.len() == product(shape)`.
- `dtype` — element type; optional on decode, `"f64"` the only accepted
  value for now.
- **Non-finite elements** spell as `{"__nonfinite__": "nan" | "inf" |
  "-inf"}` — `serde_json` has no NaN/±inf literal. Decode accepts both
  spellings, so `tensor.div` by zero and friends round-trip exactly;
  `tensor.from_list` accepts the tags and `tensor.to_list` emits them.

Engines carrying the kernel `TaggedValue::Tensor` (in `handl-value`)
decode the tag at the `call_host` boundary and carry the tensor
opaquely; engines without it see a plain object — the wire shape is the
same either way.

## Crate layout

| path | role |
|---|---|
| `src/index.hnl` | the declared op surface — the contract the Rust implements |
| `src/lib.rs` | crate root; re-exports `TENSOR_HNL`, `TENSOR_OPS`, `tensor_op_registry` |
| `src/ops/mod.rs` | `TENSOR_OPS` + `tensor_op_registry()` + the interface drift guard |
| `src/ops/codec.rs` | the `__tensor__` wire codec + `zeros`/`ones`/`from_list`/`to_list`/`shape` |
| `src/ops/elementwise.rs` | `add`/`mul`/`sub`/`div` (numpy broadcasting) + `*_backward` |
| `src/ops/matmul.rs` | `matmul` (rank-2/3, batch broadcast) + `linear` + backwards |
| `src/ops/conv.rs` | `conv2d`, `max_pool` (NCHW) + backwards |
| `src/ops/norm.rs` | `batch_norm` + backward |
| `src/ops/shape_ops.rs` | `reshape`/`unsqueeze`/`select`/`transpose`/`concat` + backwards |
| `src/ops/derive.rs` | `derive` + the `DERIV_PAIRS` adjoint table (11 pairs) |
| `src/ops/aten_generated.rs` | the generated ATen op structs (decode → call glue) |
| `src/ops/aten_kernels.rs` | the handwritten kernels behind the ATen catalog |
| `src/python.rs` | optional PyO3 binding (`--features python`) exposing `ops()`/`call()` |
| `src/tensor.test.hnl` | `.hnl`-side test corpus |

## Build and test

```bash
cargo build     # sibling [patch] tables resolve deps from ../<repo>
cargo test      # includes the index.hnl drift guard + ATen smoke tests
```

Dependencies are git deps on `branch = "main"`; the `[patch]` tables
point at sibling checkouts for local development — CI strips them and
resolves the declared git selectors.
