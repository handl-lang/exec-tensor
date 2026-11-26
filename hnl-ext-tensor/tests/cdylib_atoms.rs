//! Phase-2c e2e (P-013/proposal-0012): `libhnl_ext_tensor.so` discovered
//! as a **cdylib atom provider** and bound through
//! [`handl_plugin_lib_handl::dep_atom_modules`].
//!
//! This test binary deliberately does not link `handl-host-tensor`, so
//! no statically linked [`AtomProvider`](handl_plugin_lib_handl::atoms::AtomProvider)
//! can claim `tensor` — the cdylib fallback in
//! `plugin-lib-handl/src/cdylib_atoms.rs` is the only path that can
//! bind the dep's `atoms.yaml`. That is exactly the scenario the
//! dynamic provider exists for (hosts that cannot link the rlib).
//!
//! Requires `cargo build` first so `target/debug/libhnl_ext_tensor.so`
//! exists; the test skips cleanly otherwise.

use std::collections::HashMap;
use std::path::PathBuf;

use handl_platform::{EvalContext, Module};
use serde_json::{Value, json};

/// The built cdylib, or `None` when `cargo build` has not run.
fn tensor_cdylib() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/debug");
    for name in [
        "libhnl_ext_tensor.so",
        "libhnl_ext_tensor.dylib",
        "hnl_ext_tensor.dll",
    ] {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// `tensor.*` op call args/results use the `__tensor__` wire record —
/// the same codec `tensor_impls` and `CdylibAtomFn` share.
fn tensor(data: &[f64]) -> Value {
    json!({"__tensor__": {"shape": [data.len() as i64], "data": data, "dtype": "f64"}})
}

#[test]
fn cdylib_tensor_atoms_bind_and_dispatch() {
    let Some(so) = tensor_cdylib() else {
        eprintln!("libhnl_ext_tensor.* not built (cargo build first); skipping");
        return;
    };
    // Point ADR-0149 discovery at the cdylib's dir *before* the
    // process-wide provider scan (`cdylib_providers` is a `OnceLock`).
    // SAFETY: this test runs before any other thread in the process
    // touches the environment (single-test binary, set once at start).
    unsafe { std::env::set_var("HANDL_EXT_DIR", so.parent().unwrap()) };

    // The dep: this crate's parent repo — `atoms.yaml` + `pkg.hnl`
    // (`modules: ["tensor"]`) + the `src/index.hnl` surface.
    let dep_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("hnl-ext-tensor has a parent dir")
        .to_path_buf();
    let mut deps = HashMap::new();
    deps.insert("lib-tensor-native".to_string(), dep_dir);

    let modules = handl_plugin_lib_handl::dep_atom_modules(&deps, &HashMap::new(), "tree-walk");
    let tensor_module = modules
        .iter()
        .find(|m| m.name() == "tensor")
        .expect("dep atoms.yaml bound a `tensor` module via the cdylib provider");

    let add = tensor_module
        .function("add")
        .expect("tensor.add bound as a module function");
    let mut ctx = EvalContext::new();
    let out = add
        .call(&[tensor(&[1.0, 2.0]), tensor(&[10.0, 20.0])], &mut ctx)
        .expect("tensor.add dispatches through libhnl_ext_tensor.so");
    assert_eq!(out["__tensor__"]["data"], json!([11.0, 22.0]));

    // A second op proves the whole 239-atom surface is reachable, not
    // just the one exercised above.
    let matrix = json!({"__tensor__": {"shape": [2, 2], "data": [1.0, 2.0, 3.0, 4.0], "dtype": "f64"}});
    let shape = tensor_module
        .function("shape")
        .expect("tensor.shape bound");
    let out = shape
        .call(&[matrix], &mut ctx)
        .expect("tensor.shape dispatches");
    assert_eq!(out, json!([2, 2]));
}
