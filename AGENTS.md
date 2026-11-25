# exec-tensor-clippy

`handl-host-tensor` (clippy-clean variant) — the `tensor.*` op surface
behind the `handl_host_tensor` PyO3 extension module: a dense n-d `f64`
array (row-major) for handl runtimes. Same crate name and `pkg.hnl` as
`exec-tensor`, minus the spec/artifact machinery (`atoms.yaml`,
`host_surface.json`, `ffi.rs`, `kit.rs`, `atoms.rs`).

## Layout

| path | role |
| --- | --- |
| `src/index.hnl` | the `tensor.*` surface declarations |
| `src/ops/` | handwritten core + generated ATen catalog |
| `src/{lib,python}.rs` | crate root + PyO3 module |
| `pkg.hnl` / `pyproject.toml` | package manifest + PyO3 packaging |
| `tools/emit_tensor_language.py` | language emit helper |
| `scripts/hnl-gate.sh` | `.hnl`-source gate (typecheck + lint + test via `hnl`) |
| `tests/` | Rust tests (compose, engine roundtrip, language fragment, fixtures) |
| `docs/` | engine docs (`site.yaml`, reports) |

## Contract

- Carries the clippy-cleaned copy of the exec-tensor surface — keep it
  in sync with `exec-tensor` when porting op changes; generated ATen
  catalog comes from `tool-codegen-aten` (no in-repo generator).
- `scripts/hnl-gate.sh` needs a built `hnl` (`HNL=…` env or
  `../host-cli/target/debug/hnl`).

## Verification

```bash
cargo xtask gate                                        # fmt → clippy → test → doc
HNL=../host-cli/target/debug/hnl scripts/hnl-gate.sh   # .hnl corpus gate
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
