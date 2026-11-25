# host-tensor

Host-side tensor functions for handl runtimes — a dense n-d `f64` array
(row-major) behind the `tensor.*` op surface. Runtimes compose this
component to give `.hnl` code tensor ops; catopt is the first consumer.

The surface (239 `tensor.*` ops declared in `src/index.hnl` — the single
source of truth): a 32-op handwritten core (construction, projection,
elementwise algebra, `matmul`/`linear`, NCHW conv/pool/norm, shape ops,
and their `*_backward` adjoints) plus a ~200-op generated ATen catalog
(`src/ops/aten_generated.rs` + `aten_kernels.rs`) with a few declared
seams that fail loudly. The core ops:

| op | signature | semantics |
|---|---|---|
| `tensor.zeros` | `(List Int) -> Tensor` | zero-filled tensor of `shape` |
| `tensor.ones` | `(List Int) -> Tensor` | one-filled tensor of `shape` |
| `tensor.from_list` | `Array -> Tensor` | nested number arrays → tensor (ragged rejected) |
| `tensor.to_list` | `Tensor -> Array` | tensor → nested number arrays |
| `tensor.shape` | `Tensor -> (List Int)` | extent per axis |
| `tensor.add` | `(Tensor * Tensor) -> Tensor` | elementwise `a + b`, numpy broadcasting |
| `tensor.add_backward` | `(Tensor * Tensor * Tensor) -> Value` | adjoint of `add`: `(a, b, grad_out) -> {a, b}` — grads unbroadcast |
| `tensor.mul` | `(Tensor * Tensor) -> Tensor` | elementwise `a * b`, numpy broadcasting |
| `tensor.mul_backward` | `(Tensor * Tensor * Tensor) -> Value` | adjoint of `mul`: `da = g·b`, `db = g·a` |
| `tensor.sub` | `(Tensor * Tensor) -> Tensor` | elementwise `a − b`, numpy broadcasting |
| `tensor.sub_backward` | `(Tensor * Tensor * Tensor) -> Value` | adjoint of `sub`: `da = g`, `db = −g` |
| `tensor.div` | `(Tensor * Tensor) -> Tensor` | elementwise `a / b`, numpy broadcasting |
| `tensor.div_backward` | `(Tensor * Tensor * Tensor) -> Value` | adjoint of `div`: `da = g/b`, `db = −g·a/b²` |
| `tensor.matmul` | `(Tensor * Tensor) -> Tensor` | `(m×k)·(k×n)`; rank-3 batches over `B` (broadcast), naive loop |
| `tensor.matmul_backward` | `(Tensor * Tensor * Tensor) -> Value` | adjoint of `matmul`: `da = g·bᵀ`, `db = aᵀ·g`, unbroadcast over `B` |
| `tensor.linear` | `(Tensor * Tensor * Tensor) -> Tensor` | F.linear: `x @ wᵀ + b`; `w` is `(out, in)`, `x` rank-2/3 `(*, in)`, `b` the `(out,)` bias; no backward (`derive` → `null`) |
| `tensor.conv2d` | `Value... -> Tensor` | NCHW cross-correlation `(input, weight, bias?, stride?, padding?, groups?)` |
| `tensor.conv2d_backward` | `Value... -> Value` | adjoint of `conv2d`: `(input, weight, grad_out, bias?, stride?, padding?, groups?) -> {input, weight, bias?}` |
| `tensor.max_pool` | `Value... -> Tensor` | NCHW max pooling `(input, kernel, stride?)`; stride defaults to kernel |
| `tensor.batch_norm` | `Value... -> Tensor` | eval-mode per-channel affine `(input, gamma, beta, mean, var, eps?)` |
| `tensor.reshape` | `(Tensor * (List Int)) -> Tensor` | re-label under `shape` (product must match) |
| `tensor.unsqueeze` | `(Tensor * Int) -> Tensor` | insert a size-1 axis at `dim` (negative wraps, torch semantics); no backward (`derive` → `null`) |
| `tensor.select` | `(Tensor * Int * Int) -> Tensor` | drop axis `dim`, keep the `index`-th slice (rank −1, negative wraps); no backward (`derive` → `null`) |
| `tensor.transpose` | `(Tensor * Int * Int) -> Tensor` | swap axes `d0`/`d1` |
| `tensor.concat` | `(Array * Int) -> Tensor` | join a list of same-rank tensors along `axis` |
| `tensor.batch_norm_backward` | `Value... -> Value` | adjoint of `batch_norm`: primal args + `grad_out` → `{input, gamma, beta}` |
| `tensor.max_pool_backward` | `Value... -> Value` | adjoint of `max_pool`: `(input, grad_out, kernel, stride?, padding?)` → `{input}` — each grad scatters to its window's argmax |
| `tensor.reshape_backward` | `(Tensor * (List Int) * Tensor) -> Value` | adjoint of `reshape`: `(input, shape, grad_out)` → `{input}` |
| `tensor.transpose_backward` | `(Tensor * Int * Int * Tensor) -> Value` | adjoint of `transpose`: `(input, d0, d1, grad_out)` → `{input}` |
| `tensor.concat_backward` | `(Array * Int * Tensor) -> Value` | adjoint of `concat`: `(xs, axis, grad_out)` → `{xs}` — per-input slices |
| `tensor.derive` | `String -> Value` | registered adjoint op name for `name`, or `null` |
| `tensor.load` | `Value... -> Tensor` | `load(path[, name])` — read one tensor out of a safetensors sidecar |

**Broadcasting** — the elementwise four (`add`/`mul`/`sub`/`div`)
follow numpy rules: trailing dims align, a `1`-dim stretches, a missing
leading dim counts as `1` (`[2,3] + [3]` → `[2,3]`; `[2,1] + [1,3]` →
`[2,3]`; rank-0 is the scalar). A dim pair that differs with neither
side `1` is an error, not a coercion. `div` keeps `f64` semantics —
`x/0` is `inf`/`NaN`, matching the scalar `/`.

**Positional optionals** — `conv2d`/`max_pool`/`batch_norm` take optional trailing
parameters; a `null` (or a missing argument) selects the default, so
`conv2d(x, w, null, [2, 2])` is "no bias, stride 2×2". `conv2d` consumes
`input [N, C_in, H, W]` × `weight [C_out, C_in/groups, kH, kW]` (`bias?`
rank-1 `[C_out]`; `stride?`/`padding?` an int or `[h, w]` pair;
`groups?` a positive int — `groups = C_in = C_out` is depthwise).
`batch_norm` computes `(x − μ)/√(σ²+ε)·γ+β` per channel with rank-1 `[C]`
parameters and `eps` defaulting to `1e-5`. `max_pool` pools each channel's
`kH × kW` windows (kernel/stride an int or `[h, w]` pair; floor mode —
trailing partial windows drop). All run naive `f64` loops —
the composition seam, not BLAS.

**Shape manipulation** — `reshape(t, shape)` keeps the row-major
payload and requires the new extents' product to equal the element
count (no `-1` wildcard); `unsqueeze(t, dim)` inserts a size-1 axis
(negative `dim` counts from the end, `d % (rank+1)`);
`select(t, dim, index)` drops axis `dim` keeping the `index`-th
slice (negative `dim`/`index` wrap); `transpose(t, d0, d1)` swaps
two axes; `concat(xs, axis)` joins a non-empty list of same-rank
tensors that
agree on every axis but `axis`, whose extents sum — `axis 0` stacks,
`axis rank−1` appends along the fastest-varying axis.

**Batched `matmul`** — ranks are {2, 3} per side: `[B, m, k] ×
[B, k, n] → [B, m, n]` for rank-3, with the batch dim broadcasting (a
rank-2 operand or a `B = 1` batch repeats across the other's batches);
rank-2 × rank-2 stays the plain `(m×k)·(k×n)` product.

**Derivatives** — `tensor.derive(name)` returns the qualified op name
of the adjoint registered for `name` in the `DERIV_PAIRS` table
(qualified or bare local name; `null` when no pair exists), and the
returned name resolves through the same `ModuleRegistry` — so a `.hnl`
optimization loop can dispatch it. Eleven pairs are registered so far —
`add`/`mul`/`sub`/`div`, `matmul`, `conv2d`, `max_pool`, `reshape`,
`transpose`, `concat`, and `batch_norm`. The
`*_backward` convention: the primal's required args, then the upstream
gradient, then the primal's positional optionals; the result is the
differentiable inputs' grads keyed by parameter name —
`add_backward(a, b, g) → {"a", "b"}` (broadcast dims sum back to each
operand's shape), `matmul_backward(a, b, g) → {"a": g·bᵀ, "b": aᵀ·g}`,
`conv2d_backward(input, weight, g, bias?, stride?, padding?, groups?)
→ {"input", "weight", "bias"?}` (`db` only when `bias` was given), and
`batch_norm_backward(input, gamma, beta, mean, var, g, eps?)` →
`{"input": dx, "gamma": dγ, "beta": dβ}` (`mean`/`var` are statistical
in eval mode and carry no grad).

**`tensor.derive` is a registered-pairs table, not autodiff**
(investigated this slice). What a true `derive` would take differs by
target: for *host* ops the table is the honest minimal version — an
adjoint is another `Function` in the same registry, so `derive` is a
name lookup and the pairs are maintained by hand (the
`deriv_pairs_are_all_registered` test pins that every pair's impl
exists). For a `.hnl` *function*'s forward+backward, the host boundary
sees only a name string and evaluated argument values — `EvalContext`
can dispatch `ctx.call_function(name, …)` (so a finite-difference
gradient thunk is *possible*, but numerically approximate and O(inputs)
primal calls per grad), while an exact adjoint needs real autodiff
machinery outside this component: a source-to-source AD pass over the
`.hnl` AST (kit/language layer — the AST never crosses the `Function`
boundary) or a reverse-mode tape in the eval engine over
`Tensor`-aware primitives. Deferred; the pairs table is the surface.

**`tensor.load` — the out-of-band weight channel** (`load(path)`,
`load(path, name)`). Reads a **safetensors** file — 8-byte little-endian
header length + JSON header `{name: {dtype, shape, data_offsets}}` +
raw LE payload — and returns one tensor whose shape comes from the file
header, never the call site. `name` selects a header key; omitted, the
file must hold exactly one tensor (else the error names the keys). dtype
`"F64"` is the identity and `"F32"` widens `f32 → f64` on ingest;
`F16`/`BF16`/`I64`/`BOOL`/… fail loudly (extending is a match arm, not a
format change). Non-finite elements ride the binary channel natively and
re-encode through the `__nonfinite__` wire tags. Since ADR-0179 a tensor
with `numel >= 4096` (`handl_value::TENSOR_BIN_THRESHOLD`) skips the
decode entirely: the safetensors payload range *is* little-endian
`F32`/`F64`, so `tensor.load` returns the `__tensor_bin__` wire record
straight from the ranged read. Relative `path`
resolves against `$TENSOR_LOAD_DIR` when set — the base dir for stdin
evals (`hnl-run -` has no artifact directory, so `.hnl` +
`.safetensors` stay relocatable as a pair) — else the process cwd;
absolute paths pass through. See
`host:project/reports/tensor_load_spike.md`.

**Tensor wire value** — data on the `serde_json::Value` boundary, tagged in
the `__bytes__` style (ADR-0083 §2.1); a plain object to engines with no
`Tensor` scalar type:

```json
{"__tensor__": {"shape": [2, 3], "data": [1, 2, 3, 4, 5, 6], "dtype": "f64"}}
```

`data` elements are JSON numbers, except non-finite `f64`s — NaN,
`inf`, `-inf` — which serialize as the tagged objects
`{"__nonfinite__": "nan" | "inf" | "-inf"}` (the same `__x__` tagged
convention as `__tensor__` itself; `serde_json` has no non-finite
number literal). Decode accepts both spellings, so non-finite results
(e.g. `tensor.div` by zero) round-trip exactly; `tensor.from_list`
accepts the tags and `tensor.to_list` emits them.

**Binary `data` spelling (ADR-0179).** `data` may instead be
`{"__tensor_bin__": "<base64 little-endian payload>"}` with a required
`"dtype": "f32"|"f64"` and `payload.len() == numel × itemsize`; encoders
emit it iff `numel >= 4096` (`handl_value::TENSOR_BIN_THRESHOLD`), below
which the wire is byte-identical to the array form. Non-finite elements
ride raw IEEE bits. `decode_tensor` widens `f32 → f64` exactly (every
op still computes `Vec<f64>` — [`Tensor`] is unchanged), and a malformed
bin record is a `TypeMismatch`. Process-boundary printers must call
`handl_value::materialize_tensor_bin` before serializing so `data` is
always a list (`hnl-run` does).

## Composition

The Rust crate (`handl-host-tensor`) ships the `.hnl` interface plus the
native impls; a host wires them into its module chain:

```rust
use std::sync::Arc;
use handl_plugin_lib_handl::HandlModule;
use handl_host_tensor::{TENSOR_HNL, TENSOR_OPS, tensor_op_registry};

let module = HandlModule::new("tensor", TENSOR_HNL, &[])
    .with_ops(TENSOR_OPS, Arc::new(tensor_op_registry()));
// register `module` in the runtime's ModuleRegistry / ModuleRuntime;
// `call_host("tensor.matmul")` now reaches the Rust impl.
```

`.hnl` side (a consumer repo declares the dep in its own `.handl.toml`):

```toml
[dependencies]
host-tensor = { path = "../host-tensor/src" }
```

```hnl
from "host-tensor" import * as tensor   -- or `import tensor` (the record)
tensor.matmul(tensor.from_list([[1, 2]]), tensor.from_list([[3], [4]]))
```

## Layout

```text
src/index.hnl          the interface (cap/op/handle declarations)
src/atoms.rs           the atoms.yaml loader (P-013 spec → {fn, sig} registry)
src/ops/mod.rs         TENSOR_OPS + tensor_impls + tensor_op_registry + the interface drift guard
src/ops/codec.rs       the wire codec + boundary ops (zeros/ones/from_list/to_list/shape)
src/ops/elementwise.rs add/mul/sub/div (numpy broadcasting)
src/ops/matmul.rs      matmul (rank-2/3, batch broadcast) + linear
src/ops/conv.rs        conv2d, max_pool (NCHW)
src/ops/norm.rs        batch_norm, batch_norm_backward
src/ops/shape_ops.rs   reshape, unsqueeze, select, transpose, concat
src/ops/derive.rs      derive + DERIV_PAIRS
src/tensor.test.hnl    .hnl spec tests (host-dependent under `hnl test`)
atoms.yaml             the P-013 atom spec — GENERATED from index.hnl
host_surface.json      the tensor.* surface artifact — GENERATED by
                       tools/dump_host_surface.py (index.hnl decls +
                       tool-codegen-aten --emit-spec); Python consumers
                       (mlopt emit drift guards) vendor a pinned copy
pkg.hnl                { modules: ["tensor"] }
.handl.toml            self-declared dep names for repo-local resolution
workflows/gate.yaml                  bupha entry (handl-gates): hnl gate + cargo test
```

The `src/ops/` file map mirrors `lib-tensor`'s spec files 1:1
(`codec`/`elementwise`/`matmul`/`conv`/`norm`/`shape_ops`; `derive` splits
out of `norm` since `DERIV_PAIRS` spans domains) — the same op lives in
the same file on both sides of the pair.

## Regenerating the ATen catalog

`src/ops/aten_generated.rs` and the `ATEN-PORT-GEN` marker regions in
`src/index.hnl` / `src/ops/mod.rs` are generated by
[`tool-codegen-aten`](https://github.com/handl-lang/tool-codegen-aten)
— the ATen-schema → bindings generator extracted from this repo's old
in-repo `aten-port-gen` bin. The generator, its inputs
(`aten_port_list.txt`, `aten_schemas.json`) and the `aten_extract.py`
schema extractor all live in that repo; nothing here invokes it at
build time and the generated files stay committed. To regenerate or
drift-check:

```sh
cargo run --manifest-path ../tool-codegen-aten/Cargo.toml -- \
    --apply /path/to/lib-tensor-native           # rewrite the generated surfaces
cargo run --manifest-path ../tool-codegen-aten/Cargo.toml -- \
    --apply /path/to/lib-tensor-native --check   # drift gate, no writes
```

`~>` is the only `.hnl` arrow spelling: op-decl signatures in the
`index.hnl` marker regions use it, while the `--- aten::…` schema
comment lines keep ATen's own `->`.

## The atom spec (`atoms.yaml`, P-013/T-149)

`tensor_op_registry()` is spec-driven: `atoms.yaml` at the repo root
declares all 239 atoms `{name, sig, impl, total}` and
`src/atoms.rs` (a port of `handl-plugin-lib-stdlib`'s atoms loader —
upstream move to `handl-plugin-lib-handl` pending) builds the
`OpRegistry` `{name → {fn, sig}}` from it plus the `tensor_impls()`
table in `src/ops/mod.rs`. `impl:` keys are the `handle ... with`
bindings declared in `index.hnl`. `atoms.yaml` is generated — do not
edit by hand:

```sh
python3 tools/gen_atoms.py          # regenerate from src/index.hnl
python3 tools/gen_atoms.py --check  # drift gate, no writes
```

The `ops::tests::atoms_yaml_matches_index_hnl_and_impl_table` unit
test guards the yaml↔impls↔registry half in `cargo test`.

## The surface artifact (`host_surface.json`)

`host_surface.json` at the repo root is the checked-in snapshot of
the full `tensor.*` surface — the 32 handwritten `op` decls in
`src/index.hnl` plus the 207 ATen-ported ops' resolved spec from
`tool-codegen-aten --emit-spec` (`min_args`/`max_args`/`schema`/
`status`).  It moved here from `kit-host-tensor` (the mlopt
host-tensor backend's dissolution, P-010): this repo owns the
surface, so it owns the artifact.  Signatures are normalised to the
`->` arrow spelling.  Regenerate / drift-check:

```sh
python3 tools/dump_host_surface.py          # rewrite host_surface.json
python3 tools/dump_host_surface.py --check  # exit 1 when stale
```

Python consumers vendor a pinned copy next to their emit tables
(`mlopt.emit`'s `_OPS` drift guard); bump it deliberately, never by
hand-editing the JSON.

## Gate

```bash
cargo xtask gate  # hnl typecheck/lint/test + cargo test
cargo test                            # the Rust suite (codec + ops + dispatch)
```

## Follow-up (the catopt-shaped surface)

Deliberately out of scope so far — the composition seam is the
point, not BLAS: ~~`tensor.max_pool` and friends (NCHW layout
params)~~, ~~broadcasting `add`/`mul`/`sub`/`div`~~,
~~`reshape`/`transpose`/`concat`~~, ~~batched `matmul`~~,
~~`tensor.derive`~~ (landed as a registered-pairs table + eleven adjoints —
`.hnl`-function autodiff is the deferred half; see the finding below),
~~`TaggedValue::Tensor`~~ (landed — see the finding below), and dtype
promotion beyond `f64`/`f32` storage (ADR-0179 landed the
`F64`/`F32`-preserving `handl-value` `TensorData`; the per-op lattice
and a dtype-carrying `Tensor` here are still deferred).

**dtype promotion is a kernel-first change, then a per-op lattice.**
ADR-0179 landed the kernel half scoped to the two sidecar dtypes:
`handl-value`'s `TensorValue.data` is now `Rc<TensorData>`
(`F64`/`F32`, f64-widening equality) and the `__tensor_bin__` wire
spelling carries `"dtype": "f32"|"f64"` — but `Tensor.data` here stays
`Vec<f64>` (decode widens per call; compute is bit-identical f64). The
remaining work for a real lattice is `Tensor.data` becoming an enum
(`F64(Vec<f64>)`, `I64(Vec<i64>)`, …) and every op picking a rule per
pair — numpy-style promotion, or same-dtype-only with an error on
mixing (simpler, and what this component should probably do:
promotion lattices are where the bugs live). The odometer/index code is
dtype-agnostic; only the element type and the `f` closures change.

**`TaggedValue::Tensor` landed in `handl-value`, not in
`value_types.yaml`** (the earlier deferral, resolved). The investigation
had found `value_types.yaml` to be inert — ADR-0078 declarative spec data
that `handl-kit-data-canonical` does not load and `Language` has no
`value_types` field to merge — and the runtime value representation to be
the hardcoded `TaggedValue` enum. The kernel slice landed on that second
path: `TaggedValue::Tensor(Rc<TensorValue>)` in `handl-value` (opaque
`Rc` payload, following the `TaggedValue::Resource` precedent of
ADR-0083 §2.2), with decode/encode in the `TaggedValue`↔`Value` codec —
`call_host` decodes the `{"__tensor__": …}` wire form at the boundary and
engines carry it opaquely (`tests/engine_roundtrip.rs` pins the
round-trip on tree-walk and bytecode). The wire encoding is unchanged —
it stays the `Function`-boundary form, and engines without the variant
still see a plain object. One asymmetry remains on the *kernel* side:
`handl-value`'s `TaggedValue`↔`Value` codec still renders a
`TaggedValue::Tensor`'s non-finite elements as `null` and declines the
`__nonfinite__` tag spelling on decode — so an engine-produced tensor
carrying NaN/±inf degrades to a plain `Obj` at the boundary (the tag
objects survive verbatim inside it, and host ops fed the object decode
them exactly). Closing that half is a kernel-repo change.

## License

MIT.

## Documentation

<https://handl-lang.github.io/docs/host-tensor/>
