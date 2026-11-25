#!/usr/bin/env python3
"""Generate `atoms.yaml` from `src/index.hnl` (P-013/T-149).

`src/index.hnl` is the single source of truth for the `tensor.*` surface:
each op is declared three times — `cap "tensor.x"`, `op "tensor.x" : Sig`
(with an indented `guarantee:`/`requires:`/`termination:` clause block),
and `handle "tensor.x" with <impl-key>`. This script folds the three
decl forms into the atom IDL row:

    - name  <- the quoted op name        (`tensor.abs`)
    - sig   <- the `op` decl's signature (`Tensor ~> Tensor`)
    - impl  <- the `handle` decl's `with` binding (`rust_abs`)
    - total <- a `guarantee: True` clause is present
    - requires / termination / guarantee <- every indented decl clause
      verbatim (T-151 caps-as-data): repeated clause keys accumulate,
      nothing is dropped. `guarantee: True` itself stays the `total`
      flag and is not duplicated into `guarantee:`.

`caps:` is deliberately NOT generated: index.hnl decls carry no
capability-requirement syntax today, so the field is hand-authored spec
data (consumed by `pkg.hnl` dep-scope `caps:` filtering in
`handl-plugin-lib-handl`). If a `cap`-level clause syntax lands, extend
`parse_index` to emit it.

`atoms.yaml` sits at the package root next to `pkg.hnl` — the same
vendoring convention `handl-pkg-vendor` applies to `pkg.hnl`.

Usage:
    python3 tools/gen_atoms.py          # (re)write atoms.yaml
    python3 tools/gen_atoms.py --check  # drift guard: exit 1 on mismatch
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INDEX = REPO / "src" / "index.hnl"
OUT = REPO / "atoms.yaml"
MODULE = "tensor"

# `op "tensor.abs" : Tensor ~> Tensor` — the sig is the rest of the line.
RE_OP = re.compile(r'^op\s+"([^"]+)"\s*:\s*(.*?)\s*$')
RE_CAP = re.compile(r'^cap\s+"([^"]+)"')
RE_HANDLE = re.compile(r'^handle\s+"([^"]+)"\s+with\s+(\S+)')
RE_CLAUSE = re.compile(r"^\s+(guarantee|requires|termination)\s*:\s*(.*)$")

HEADER = """\
# exec-tensor atom spec (P-013/T-149) — GENERATED, do not edit by hand.
# Source of truth: src/index.hnl (`op`/`guarantee`/`handle` decls).
# Regenerate: python3 tools/gen_atoms.py   Drift check: --check
# Schema matches handl-plugin-lib-handl's atoms.yaml: `name` is the
# dispatch key, `sig` the handl signature (~> canonical), `impl` the key
# into the Rust impl table (`src/ops/mod.rs` `tensor_impls()`), `total`
# the `guarantee: True` marker, and `requires`/`termination`/`guarantee`
# the verbatim decl clauses (T-151). `caps` is hand-authored, not
# generated — index.hnl has no capability-requirement syntax.
module: tensor
atoms:
"""


def parse_index(text: str) -> list[dict]:
    """Fold cap/op/handle decls into atom rows, in declaration order."""
    caps: list[str] = []
    ops: dict[str, str] = {}
    op_order: list[str] = []
    # name -> clause key -> [texts]; every clause line is kept (T-151 —
    # the previous last-wins dict silently dropped repeats).
    clauses: dict[str, dict[str, list[str]]] = {}
    handles: dict[str, str] = {}

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if m := RE_CAP.match(line):
            caps.append(m.group(1))
        elif m := RE_OP.match(line):
            name, sig = m.group(1), m.group(2)
            ops[name] = sig
            op_order.append(name)
            # Slurp the indented clause block (guarantee:/requires:/...).
            clause: dict[str, list[str]] = {}
            j = i + 1
            while j < len(lines):
                cm = RE_CLAUSE.match(lines[j])
                if not cm:
                    break
                clause.setdefault(cm.group(1), []).append(cm.group(2).strip())
                j += 1
            clauses[name] = clause
            i = j - 1
        elif m := RE_HANDLE.match(line):
            handles[m.group(1)] = m.group(2)
        i += 1

    missing = [n for n in op_order if n not in handles]
    if missing:
        raise SystemExit(f"index.hnl: ops without handle decl: {missing}")
    orphans = set(handles) - set(ops)
    if orphans:
        raise SystemExit(f"index.hnl: handle decls without op decl: {sorted(orphans)}")
    cap_set = set(caps)
    op_set = set(op_order)
    if cap_set != op_set or len(caps) != len(op_order):
        raise SystemExit(
            f"index.hnl: cap/op decl mismatch "
            f"(caps-only={sorted(cap_set - op_set)}, ops-only={sorted(op_set - cap_set)})"
        )

    atoms = []
    for name in op_order:
        clause = clauses.get(name, {})
        guarantees = clause.get("guarantee", [])
        atoms.append(
            {
                "name": name,
                "sig": ops[name],
                "impl": handles[name],
                "total": "True" in guarantees,
                # `guarantee: True` stays the `total` flag — not duplicated.
                "guarantee": [g for g in guarantees if g != "True"],
                "requires": clause.get("requires", []),
                "termination": clause.get("termination", []),
            }
        )
    return atoms


def render(atoms: list[dict]) -> str:
    out = [HEADER]
    for a in atoms:
        out.append(f'  - name: {a["name"]}\n')
        out.append(f'    sig: {json.dumps(a["sig"])}\n')
        out.append(f'    impl: {a["impl"]}\n')
        if a["total"]:
            out.append("    total: true\n")
        # T-151: verbatim decl clauses, emitted only when present.
        for key in ("requires", "termination", "guarantee"):
            vals = a.get(key) or []
            if vals:
                out.append(f"    {key}:\n")
                for v in vals:
                    out.append(f"      - {json.dumps(v)}\n")
    return "".join(out)


def main() -> int:
    atoms = parse_index(INDEX.read_text())
    text = render(atoms)
    if "--check" in sys.argv[1:]:
        cur = OUT.read_text() if OUT.exists() else ""
        if cur != text:
            print(f"{OUT}: stale — run `python3 tools/gen_atoms.py`", file=sys.stderr)
            return 1
        print(f"{OUT}: up to date ({len(atoms)} atoms)")
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(atoms)} atoms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
