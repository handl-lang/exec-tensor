# host-tensor

Host-side `tensor.*` functions: a dense n-d `f64` array (row-major)
behind the `tensor.*` op surface, for `.hnl` runtimes to compose.

## What it is

Runtimes compose this component to give `.hnl` code tensor operations.
A tensor is **data on the `serde_json::Value` boundary**, not a language
feature: it crosses host calls as a tagged wire object
(`{"__tensor__": {"shape": [...], "data": [...], "dtype": "f64"}}`), so
engines with no tensor scalar see an ordinary object while engines
carrying the kernel `TaggedValue::Tensor` decode the tag at the
`call_host` boundary and pass it opaquely.

`src/index.hnl` is the single source of truth for the surface — 239
`tensor.*` ops declared as FFI atoms — and the Rust side in `src/ops/`
implements exactly what that file declares:

- a **handwritten core** (32 ops): construction and projection
  (`zeros`, `ones`, `from_list`, `to_list`, `shape`), elementwise
  algebra with numpy broadcasting (`add`/`mul`/`sub`/`div`), `matmul`
  and `linear`, NCHW `conv2d`/`max_pool`/`batch_norm`, shape
  manipulation (`reshape`, `unsqueeze`, `select`, `transpose`,
  `concat`), and the `*_backward` adjoints `tensor.derive` resolves;
- a **generated ATen catalog** (207 ops): the PyTorch `aten::*`
  vocabulary — unary math and activations, comparisons, reductions,
  indexing, linalg, constructors — ported in `src/ops/aten_generated.rs`
  with kernels in `aten_kernels.rs`. A few entries are declared **seams**
  that fail loudly with an honest error until a handwritten kernel
  lands (RNG ops like `rand`/`normal`, uninitialized `empty`).

All ops are FFI atoms: a tensor has no literal syntax and no
eval-vocabulary constructor, so tensor code cannot be expressed as pure
handl arrows — the `op`/`handle` declaration form declares the
interface and the Rust `Function` impls provide the logic.

## Wire format

```json
{"__tensor__": {"shape": [2, 3], "data": [1, 2, 3, 4, 5, 6], "dtype": "f64"}}
```

`shape` is the extent per axis, `data` the flat row-major payload
(`data.len() == product(shape)`), `dtype` the element type (`"f64"` is
the only value). Non-finite elements — NaN, `inf`, `-inf` — spell as
`{"__nonfinite__": "nan" | "inf" | "-inf"}` because JSON has no non-finite
number literal; decode accepts both spellings, so results like
`tensor.div` by zero round-trip exactly.

## Modules

- `src/index.hnl` — the declared op surface (the contract the Rust implements)
- `src/ops/` — the Rust implementations (codec, elementwise, matmul,
  conv, norm, shape ops, derive, and the generated ATen catalog)
- `src/tensor.test.hnl` — the `.hnl` test corpus

## Start here

- **Reference** — [public API and the op catalog](/host-tensor/reference/)
- **How-to** — [task recipes](/host-tensor/how-to/)
- **Explanation** — [design and rationale](/host-tensor/explanation/)

No tutorials: this repo is a building block used from Rust. End-to-end
walkthroughs live with the product that composes it — see the
[Handl platform](/handl/) and [graphopt](/graphopt/).
