# ADR-0174 P1 — the tensor `Language` fragment (ops/attrs/guards as data)

Phase 1 of the end-state map: tensor ops stop being *only* Python/Rust
tables and `.hnl` `op`/`cap` decls, and become `handl-language`
fragment data a `Language` can merge (ADR-0044). This is the phase
that makes "ops are data" real for the tensor domain — **data +
generator + parse-level proof only**; no execution, no migration of
`mlopt_core.ir`, no changes to the five Python tables (P3).

## Where it lives

`host-tensor/data/language/tensor/ast.yaml` — generated, committed,
drift-guarded. Rationale:

- **host-tensor owns the op surface.** The two authoritative sources
  (`src/index.hnl` handwritten `op`/`cap` decls and
  `aten_port_gen --emit-spec` for the 207 ATen-ported ops) are both in
  this repo; the fragment is generated from them directly, not from
  graphopt's `host_surface.json` copy.
- **ADR-0175 dissolution.** The map says `host-tensor → lib-tensor +
  tensor Language fragment data`. This file is exactly that data;
  when host-tensor dissolves it moves with the op decls. It is
  portable YAML — no repo-local machinery is required to consume it.
- **kit-data-prelude / kernel `data/` were rejected.** Those hold the
  *base* language (the shell every domain overlays); tensor ops are a
  domain fragment, merged on top — same posture as graphopt's
  `vocab/ast.yaml` projection and the `catopt-fragments` test
  fixtures.
- The sibling meta-tier kit work (kit-prelude*) was checked first —
  no file conflicts; nothing here touches `kit-*`.

## What it declares

One `AstShapeDef` per `tensor.*` op (238 rows = 31 handwritten seam
ops + 207 ATen-ported ops), the same op→fragment encoding
`discoverykit.handl.opdef_to_handl` / `vocabdir._ast_shape_row`
already use for mlopt ops:

- `node_kind` — the qualified op name (`tensor.abs`).
- `children` — `arg{i}` roles for ops with known arity:
  required through `min_args`, `optional: true` through `max_args`
  (the positional-optional convention — `conv2d`'s trailing
  `bias?/stride?/padding?/groups?`). Ops whose arity the surface
  leaves open (the 30 handwritten decls carry no `min_args`/`max_args`)
  emit **no** `children` — the `AstShapeDef` opaque/positional
  fallback, not a guessed arity.
- `payload_schema` — a real JSON Schema for the node payload
  (`attrs` object with the declared spellings as properties,
  `additionalProperties: true`; `children` array with
  `minItems`/`maxItems` when known) **plus** a `tensor_op` extension
  block carrying the full surface row (`signature`, `min_args`,
  `max_args`, `status`, `kind`, ATen `schema`, `reason`) and the
  attrs schema.

### Attrs as data

`tensor_op.attrs_schema` per op, derived from
`mlopt_core.attrs.ATTR_SCHEMA`/`ATTR_REQUIRED` (eqsat) keyed by the
bare op name:

- `positions` — arg-position → canonical attr name (`{"1": "dim"}`).
- `names` — every accepted spelling: canonical names **and** the
  declared `argN` positional aliases (`["arg1", "dim"]`).
- `required` — `ATTR_REQUIRED` names. Recorded as *data*, not as
  unconditional JSON-Schema `required`: mlopt enforces requiredness
  only on fully-attributed terms (a term carrying ≥1 declared attr),
  so zero-attr partial/pattern terms stay legal — the schema says so
  via `open: true` rather than lying.
- Ops absent from `ATTR_SCHEMA` get `source: null`, empty
  `names`/`required`, `open: true` — honestly open attrs, matching
  `Op.make(validate=False)` semantics.

### Guards as data

The guard *vocabulary* already exists as data —
`TransformCondition::{AttrEq, SameAttr, KindAt, ChildCount, …}`
(kernel `handl-language`, ADR-0094/0173 Phase B). The fragment
supplies what guards range over: per-op declared attr names,
positions, and requiredness, plus arity bounds (`minItems`/
`maxItems`, `optional` roles). No `transforms.yaml` rows are emitted
in P1 — tensor laws are not migrated here (they live in graphopt's
`vocab/transforms.yaml` until P3).

## Generation

```sh
python3 tools/emit_tensor_language.py          # write data/language/tensor/ast.yaml
python3 tools/emit_tensor_language.py --check  # drift gate (exit 1 when stale)
```

Inputs: `src/index.hnl` op decls, `target/debug/aten_port_gen
--emit-spec` (build first: `cargo build --bin aten_port_gen`), and
`mlopt_core.attrs` — resolved by importing `mlopt_core.attrs` with
`MLCORE_SRC`/`EQSAT_DIR` or the `../eqsat` sibling checkout on
`sys.path`. Requires PyYAML. Generated artifacts win on name clashes
(same rule as `mlopt_host_tensor.surface.collect_host_surface`).

**Known upstream drift:** graphopt's `host_surface.json` carries 237
ops because `mlopt_host_tensor.surface._OP_DECL` requires the result
type to be a single `\S+` token — `tensor.shape : Tensor -> (List
Int)` never matches. This fragment's regex takes the rest of the
line, so it covers the full 238-op surface. Worth fixing
`surface.py`'s regex (or regenerating its artifact) separately.

## Proof

`tests/language_fragment.rs` (cargo test, no engine execution):

- all 238 rows deserialize as `AstShapeDef` and merge into a
  `Language` via `merge_ast_fragment`;
- attrs schema round-trips out of `payload_schema.tensor_op`
  (`tensor.concat` → `positions {"1": "dim"}`, `names [arg1, dim]`,
  `required [dim]`; `tensor.abs` → honestly open);
- arity declarations verified (fixed `arg0`; optional `arg2` on
  `constant_pad_nd`; open arity on `conv2d` emits no roles);
- `Term::op_native_attrs("tensor.concat", …, {"dim": -1})` survives a
  `serde_json` round-trip with attrs intact and is admitted by the
  declared schema (canonical and `argN` spellings).

## Deferred (explicitly not P1)

- **Execution/binding** — ops are data; nothing evaluates them. The
  `OpRegistry`/`index.hnl` decls remain the runtime surface until a
  later phase binds `Language` op declarations to host impls.
- **`Language.ops` loader** — `Language.ops` is a
  `BTreeMap<String, Value>` populated programmatically today; there is
  no `ops.yaml` fragment kind in `language_from_dir`. If engines want
  the evaluated-op-decl form (`{"__tag__": "op", …}`), a loader is a
  small kernel/kit-prelude-lang addition — deferred.
- **`transforms.yaml`/laws** — tensor law guards (AttrEq/SameAttr
  conditions) are vocabulary-ready; emitting tensor-domain laws is
  P3 table-migration work.
- **`types.yaml` rules** — per-op typecheck `Term` schemes are
  hand-authored kernel data today; generating them is a later phase.
- **mlopt-side adoption** — the five Python tables
  (`emit._OPS`, `hnl_surface._SPECS`, `HOST_BINDINGS`,
  `torch_bindings`, `modelproj`) still stand; this fragment is the
  target they consolidate onto, not a replacement yet.
- **`host_surface.json` regex fix** — see "Known upstream drift".
