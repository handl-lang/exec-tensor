#!/usr/bin/env python3
"""Generate ``host_surface.json`` — the ``tensor.*`` op surface artifact.

The authoritative ``tensor.*`` surface lives in this repo:

* the handwritten ``op "tensor.X" : <signature>`` declarations in
  ``src/index.hnl`` (the seam ops the Rust impls own);
* ``tool-codegen-aten --emit-spec`` — the resolved spec for the
  ATen-ported ops (``handl-lang/tool-codegen-aten``'s
  ``tools/aten_port_list.txt`` + ``tools/aten_schemas.json`` →
  ``src/ops/aten_generated.rs``), carrying
  ``min_args``/``max_args``/``schema``/``status``.

``host_surface.json`` at the repo root is the checked-in snapshot —
generated, do not hand-edit (same convention as ``atoms.yaml``).
Python consumers (``mlopt.emit``'s drift guard) vendor a pinned copy;
this repo owns freshness.

Every entry is ``{"op": "tensor.<name>", "signature": …,
"min_args": …, "max_args": …|null, "status": …, "kind":
"handwritten"|"generated"}``.  Signatures are normalised to the
``->`` arrow spelling (``index.hnl``/the generator emit ``~>``) so
the artifact schema is stable for consumers that split on ``->``.

Usage::

    python3 tools/dump_host_surface.py          # (re)write host_surface.json
    python3 tools/dump_host_surface.py --check  # drift guard: exit 1 on mismatch

``TOOL_CODEGEN_ATEN`` overrides the tool-codegen-aten binary path;
the default is the sibling ``../tool-codegen-aten`` checkout's
``target/<profile>/`` build.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
SURFACE_JSON = REPO / "host_surface.json"

#: ``op "tensor.conv2d" : Value... -> Tensor`` (or ``~>`` — both arrow
#: spellings appear across the ``->``/``~>`` migration; the artifact
#: normalises to ``->``).  The result side is ``(.+?)$`` (rest of
#: line), NOT ``\\S+$``: ``tensor.shape`` returns ``(List Int)`` — a
#: two-token composite type a single-token pattern silently drops.
_OP_DECL = re.compile(
    r'^op "(tensor\.[A-Za-z0-9_]+)"\s*:\s*(.+?)\s*(?:->|~>)\s*(.+?)\s*$',
    re.MULTILINE,
)


def _norm_arrow(sig: str) -> str:
    """Normalise the ``~>`` arrow spelling to ``->`` for the artifact."""
    return sig.replace("~>", "->")


def _tool_codegen_aten() -> tuple[Path, Path]:
    """Locate the ``tool-codegen-aten`` binary and its repo root.

    Returns ``(binary, tool_repo)``.  Resolution order:
    ``$TOOL_CODEGEN_ATEN`` (explicit binary path — the tool repo is
    its parent's parent), then the sibling-checkout convention
    ``../tool-codegen-aten/target/{release,debug}/``.
    """
    env = os.environ.get("TOOL_CODEGEN_ATEN")
    if env:
        binary = Path(env)
        repo = binary.resolve().parents[2]
        if binary.is_file() and (repo / "tools" / "aten_port_list.txt").is_file():
            return binary, repo
        raise FileNotFoundError(
            f"TOOL_CODEGEN_ATEN={env} is not a tool-codegen-aten "
            "binary inside a tool-codegen-aten checkout"
        )
    repo = REPO.parent / "tool-codegen-aten"
    for profile in ("release", "debug"):
        binary = repo / "target" / profile / "tool-codegen-aten"
        if binary.is_file():
            return binary, repo
    raise FileNotFoundError(
        f"tool-codegen-aten binary not found under {repo}/target — "
        "build it: cargo build --manifest-path "
        "../tool-codegen-aten/Cargo.toml --bin tool-codegen-aten "
        "(or set TOOL_CODEGEN_ATEN)"
    )


def _index_hnl_ops() -> dict[str, dict[str, Any]]:
    """Handwritten ``op "tensor.X" : sig`` decls from ``index.hnl``."""
    text = (REPO / "src" / "index.hnl").read_text()
    ops: dict[str, dict[str, Any]] = {}
    for m in _OP_DECL.finditer(text):
        name, params, result = m.group(1), m.group(2), m.group(3)
        ops[name] = {
            "op": name,
            "signature": _norm_arrow(f"{params.strip()} -> {result.strip()}"),
            "min_args": None,
            "max_args": None,
            "status": "Handwritten",
            "kind": "handwritten",
            "reason": "",
        }
    return ops


def _emit_spec() -> list[dict[str, Any]]:
    """Run ``tool-codegen-aten --emit-spec``; return its ``ops`` array."""
    binary, tool_repo = _tool_codegen_aten()
    out = subprocess.run(
        [
            str(binary),
            "--schemas",
            str(tool_repo / "tools" / "aten_schemas.json"),
            "--list",
            str(tool_repo / "tools" / "aten_port_list.txt"),
            "--emit-spec",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"tool-codegen-aten --emit-spec failed: {out.stderr.strip()}"
        )
    return json.loads(out.stdout)["ops"]


def collect_host_surface() -> list[dict[str, Any]]:
    """Build the unified op→signature table.

    Generated (ATen-ported) entries win over same-named handwritten
    decls — the generator owns those names.  Sorted by op name for a
    stable artifact.
    """
    ops = _index_hnl_ops()
    for entry in _emit_spec():
        name = entry["op"]
        ops[name] = {
            "op": name,
            "signature": _norm_arrow(str(entry.get("signature", ""))),
            "schema": entry.get("schema", ""),
            "min_args": entry.get("min_args"),
            "max_args": entry.get("max_args"),
            "status": entry.get("status", ""),
            "kind": "generated",
            "reason": entry.get("reason", ""),
        }
    return sorted(ops.values(), key=lambda o: o["op"])


def render_surface(ops: list[dict[str, Any]]) -> str:
    """Canonical artifact text — what this script writes."""
    doc = {
        "provenance": (
            "Generated by tools/dump_host_surface.py from this repo "
            "(src/index.hnl op decls + tool-codegen-aten --emit-spec). "
            "Do not hand-edit; regenerate."
        ),
        "ops": ops,
    }
    return json.dumps(doc, indent=2, sort_keys=False) + "\n"


def main() -> int:
    """Collect the surface; write or check the artifact."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare against the artifact without writing",
    )
    args = ap.parse_args()

    ops = collect_host_surface()
    text = render_surface(ops)

    if args.check:
        current = SURFACE_JSON.read_text() if SURFACE_JSON.is_file() else ""
        if current != text:
            print(
                f"{SURFACE_JSON} is stale — run "
                "python3 tools/dump_host_surface.py",
                file=sys.stderr,
            )
            return 1
        print(f"{SURFACE_JSON} is fresh ({len(ops)} ops)")
        return 0

    SURFACE_JSON.write_text(text)
    print(f"wrote {SURFACE_JSON} ({len(ops)} ops)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
