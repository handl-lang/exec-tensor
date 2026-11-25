# lib-tensor-native

`handl-host-tensor` — host-side tensor functions for handl runtimes: a
dense n-d `f64` array (row-major) behind the `tensor.*` op surface.
Runtimes compose this component to give `.hnl` code tensor ops.

## Layout

| path | role |
| --- | --- |
| `src/index.hnl` | the `tensor.*` surface — single source of truth (239 op decls) |
| `src/ops/` | handwritten core + generated ATen catalog (`aten_generated.rs`, `aten_kernels.rs`) |
| `src/{lib,ffi,kit,python}.rs` | crate root, FFI, kit wiring, PyO3 module |
| `atoms.yaml` | the P-013 atom spec — GENERATED from `index.hnl`, do not hand-edit |
| `host_surface.json` | checked-in `tensor.*` surface artifact — GENERATED |
| `tools/` | `gen_atoms.py`, `dump_host_surface.py`, `emit_tensor_language.py` |
| `pkg.hnl` | package manifest (`modules: ["tensor"]`) |
| `pyproject.toml` | `handl_host_tensor` PyO3 extension |
| `scripts/hnl-gate.sh` | `.hnl`-source gate (typecheck + lint + test via `hnl`) |
| `tests/` | Rust + `.hnl` engine tests |

## Contract

- 32-op handwritten core (construction, projection, elementwise algebra,
  `matmul`/`linear`, NCHW conv/pool/norm, shape ops, `*_backward`
  adjoints) + ~200-op generated ATen catalog with declared seams that
  fail loudly.
- `atoms.yaml` is generated: `python3 tools/gen_atoms.py` (regen) /
  `--check` (drift gate, no writes).
- Broadcasting follows numpy rules on the elementwise four
  (`add`/`mul`/`sub`/`div`); `div` keeps `f64` semantics (`x/0` →
  `inf`/`NaN`).
- `TaggedValue::Tensor` lives in `handl-value` (kernel repo); the wire
  form is `{"__tensor__": …}` — non-finite elements degrade to `null`
  at the kernel codec boundary (documented asymmetry).

## Verification

```bash
cargo xtask gate                                  # fmt → clippy → test → doc
HNL=../host-cli/target/debug/hnl scripts/hnl-gate.sh   # .hnl corpus gate
python3 tools/gen_atoms.py --check                # atoms.yaml drift gate
```

<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->
