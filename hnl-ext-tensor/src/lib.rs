//! `hnl-ext-tensor` — the `tensor.*` host-op surface as a dynamic `hnl`
//! extension (ADR-0149).
//!
//! Two jobs:
//!
//! 1. **Kit registration** — [`handl_host_tensor::kit::register`] seats
//!    the `tensor` [`handl_plugin_lib_handl::HandlModule`] (the authored
//!    `tensor.hnl` surface + all native ops from
//!    [`handl_host_tensor::tensor_op_registry`]) into
//!    `handl-kit-core`'s process-wide module table. `handl-kit-prelude`'s
//!    canonical kit calls it from its own `register()`, which is what
//!    makes `hnl run file.hnl` execute `tensor.*` ops natively; calling
//!    it here too keeps this extension self-sufficient.
//!
//! 2. **`tensor-run` command** — a self-contained `.hnl` runner for
//!    environments where the `eval` extension predates the tensor kit
//!    wiring: `hnl tensor-run file.hnl` discovers the project kit
//!    (`.handl.toml` deps, module root), wraps the source in `do { … }`,
//!    and evaluates through [`Kit::eval`]. Output is the result value
//!    with `__tensor_bin__` payloads materialized back to `data` lists —
//!    the same egress `hnl run` performs, minus arrow-term pretty
//!    printing (raw JSON for non-string values).
//!
//! 3. **cdylib atom provider** (P-013 phase-2c, proposal-0012) — the
//!    manifest carries `atoms_yaml` (the repo's generated `atoms.yaml`),
//!    marking this `.so` as an atom provider for `tensor.*`. When a
//!    project dep on `lib-tensor-native` binds its `atoms.yaml` and no
//!    statically linked provider claims `tensor`,
//!    `handl-plugin-lib-handl`'s dep binding discovers this library
//!    through the ADR-0149 search path and marshals each atom call over
//!    `handl_ext_dispatch` (`cmd` = `"tensor.<op>"`, `args[0]` = JSON
//!    args array). [`dispatch`] routes those calls through
//!    [`handl_host_tensor::ffi`] — the same `AtomHandle` path
//!    `ffi.rust.resolve` uses — so args/results keep the `__tensor__`
//!    wire codec and get a fresh `EvalContext` (no host recursion — the
//!    phase-2c cdylib limit).
//!
//!    Statically linked providers (the `inventory` `AtomProvider` in
//!    `handl-host-tensor`'s `kit.rs`, anchored by `handl-kit-prelude`)
//!    always win over this cdylib — the dynamic path exists for hosts
//!    that cannot link the rlib.
//!
//! `tensor` must be admitted by the prelude manifest's `base_modules`
//! (it is, in `handl-kit-data-prelude`'s `prelude.yaml`) — the kit
//! registry filter only materializes listed modules.

use std::path::{Path, PathBuf};

use handl_ext::{ExtRequest, ExtResponse, ExtensionManifest};
use handl_kit_pkg::KitProjectExt;
use handl_kit_prelude::Kit;
use serde_json::Value;

handl_ext::export!(
    ExtensionManifest::new("tensor")
        .version(env!("CARGO_PKG_VERSION"))
        .command(
            "tensor-run",
            "tensor-run [--kit <dir>] <file.hnl> — run a .hnl program with the tensor.* host surface",
        )
        .command("tensor-ops", "tensor-ops — list the registered tensor.* host ops")
        // Atom-provider claim surface (P-013 phase-2c): the generated
        // spec doubles as the manifest's `atoms_yaml` — hosts parse it
        // for the `tensor` module claim; the bound spec still comes from
        // the consuming dep's own `atoms.yaml`.
        .atoms_yaml(include_str!("../../atoms.yaml")),
    dispatch
);

/// ABI dispatch: `tensor.*` atom calls, `tensor-run` executes a file,
/// `tensor-ops` lists ops.
fn dispatch(cmd: &str, req: &ExtRequest) -> ExtResponse {
    // Seat the `tensor` module before any kit assembly reads the table —
    // also what `ffi::resolve` consults for atom calls.
    handl_host_tensor::kit::register();
    // Atom names are dispatched verbatim and never appear in `commands`;
    // every `tensor.<op>` wire lands here (P-013 phase-2c).
    if cmd.starts_with("tensor.") {
        return tensor_atom(cmd, req);
    }
    match cmd {
        "tensor-run" => tensor_run_command(&req.args, Path::new(&req.cwd)),
        "tensor-ops" => tensor_ops_command(),
        other => ExtResponse::error(2, format!("tensor: unknown command `{other}`\n")),
    }
}

/// Serve one `tensor.*` atom call: `args[0]` is the JSON args array;
/// the result is the op's `serde_json::Value` reply (`__tensor__` wire
/// records included). Resolution goes through the kit module table via
/// [`handl_host_tensor::ffi`] — the same handle path `ffi.rust.resolve`
/// uses — with a fresh `EvalContext` per call.
fn tensor_atom(cmd: &str, req: &ExtRequest) -> ExtResponse {
    let args = match handl_ext::atom_call_args(req) {
        Ok(a) => a,
        Err(r) => return r,
    };
    let result = handl_host_tensor::ffi::resolve(cmd)
        .and_then(|handle| handl_host_tensor::ffi::call(&handle, &args));
    match result {
        Ok(value) => handl_ext::atom_result(&value),
        Err(e) => handl_ext::atom_failure(format!("tensor: {cmd}: {e}")),
    }
}

/// `tensor-ops` — the declared `tensor.*` op names, one per line.
fn tensor_ops_command() -> ExtResponse {
    let mut out = String::new();
    for op in handl_host_tensor::TENSOR_OPS {
        out.push_str(op);
        out.push('\n');
    }
    ExtResponse::ok(out)
}

/// `tensor-run [--kit <dir>] <file.hnl>`.
fn tensor_run_command(args: &[String], cwd: &Path) -> ExtResponse {
    let mut kit_dir: Option<PathBuf> = None;
    let mut file: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--kit" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return ExtResponse::error(2, "tensor-run: --kit requires a directory\n");
                };
                kit_dir = Some(resolve(cwd, v));
            }
            other if other.starts_with('-') => {
                return ExtResponse::error(
                    2,
                    format!("tensor-run: unknown flag `{other}`\n"),
                );
            }
            other => {
                if file.is_some() {
                    return ExtResponse::error(
                        2,
                        "tensor-run: expected exactly one <file.hnl>\n",
                    );
                }
                file = Some(resolve(cwd, other));
            }
        }
        i += 1;
    }
    let Some(file) = file else {
        return ExtResponse::error(
            2,
            "usage: tensor-run [--kit <dir>] <file.hnl>\n",
        );
    };

    let src = match std::fs::read_to_string(&file) {
        Ok(s) => s,
        Err(e) => {
            return ExtResponse::error(1, format!("tensor-run: {file:?}: {e}\n"));
        }
    };

    let kit = match &kit_dir {
        Some(kd) => Kit::discover_with_kit(&file, kd),
        None => Kit::discover(&file),
    };
    let kit = match kit {
        Ok(k) => k,
        Err(e) => return ExtResponse::error(1, format!("tensor-run: {e}\n")),
    };

    match kit.eval(&wrap_run_source(&src)) {
        Ok(mut value) => {
            // ADR-0179 egress: expand `__tensor_bin__` payloads back to
            // `data` lists before printing (same step `hnl run` performs).
            handl_value::materialize_tensor_bin(&mut value);
            let out = match &value {
                Value::String(s) => format!("{s}\n"),
                other => format!(
                    "{}\n",
                    serde_json::to_string_pretty(other).unwrap_or_default()
                ),
            };
            ExtResponse::ok(out)
        }
        Err(e) => ExtResponse::error(1, format!("tensor-run: {e}\n")),
    }
}

/// Resolve `path` (possibly relative) against the request's `cwd`.
fn resolve(cwd: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    }
}

/// Wrap `source` in `do { … }` so a multi-statement file evaluates as one
/// block whose tail is the result — the same wrapping `hnl run` applies.
/// A source already leading with `do` is used verbatim.
fn wrap_run_source(source: &str) -> String {
    let trimmed = source.trim_start();
    if trimmed.starts_with("do") {
        source.to_string()
    } else {
        format!("do {{\n{source}\n}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_run_source_variants() {
        assert_eq!(wrap_run_source("x := 1"), "do {\nx := 1\n}");
        assert_eq!(wrap_run_source("do { x := 1 }"), "do { x := 1 }");
        assert_eq!(wrap_run_source("  do { x := 1 }"), "  do { x := 1 }");
    }

    #[test]
    fn tensor_ops_lists_every_op() {
        let res = tensor_ops_command();
        assert_eq!(res.code, 0);
        assert!(res.stdout.contains("tensor.from_list"));
        assert!(res.stdout.contains("tensor.matmul"));
    }

    /// Phase-2c atom wire: `tensor.add` arrives as `cmd` with `args[0]`
    /// a JSON array of `__tensor__` records; the reply is the op's JSON
    /// result. Exercises the same path `CdylibAtomFn` drives.
    #[test]
    fn tensor_atom_dispatch_roundtrips() {
        use serde_json::json;
        let tensor = |data: &[f64]| {
            json!({"__tensor__": {"shape": [data.len() as i64], "data": data, "dtype": "f64"}})
        };
        let payload = serde_json::to_string(&json!([tensor(&[1.0, 2.0]), tensor(&[10.0, 20.0])]))
            .unwrap();
        let req = ExtRequest {
            args: vec![payload],
            ..ExtRequest::default()
        };
        let res = dispatch("tensor.add", &req);
        assert_eq!(res.code, 0, "stderr: {}", res.stderr);
        let out: Value = serde_json::from_str(&res.stdout).unwrap();
        assert_eq!(out["__tensor__"]["data"], json!([11.0, 22.0]));

        // Unknown atoms fail loudly through the same wire.
        let req = ExtRequest {
            args: vec!["[]".to_string()],
            ..ExtRequest::default()
        };
        let res = dispatch("tensor.no_such_op", &req);
        assert_ne!(res.code, 0);
    }
}
