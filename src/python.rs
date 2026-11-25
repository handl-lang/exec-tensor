//! Python binding for the `tensor.*` op surface — the PyO3 half of the
//! Phase-1 host-tensor sink (graphopt `docs/reports/host_tensor_sink_phase0.md`).
//!
//! Build with `cargo build --features python`; the cdylib artifact
//! (`target/<profile>/libhandl_host_tensor.so`) imports as the
//! `handl_host_tensor` extension module once renamed
//! (`handl_host_tensor.so`).  The installable path is maturin —
//! `maturin build` (see `pyproject.toml`) produces a wheel whose
//! `handl_host_tensor` package wraps this module, built with
//! `pyo3/extension-module`.  The surface is deliberately thin — the
//! `Function` contract is already JSON-in/JSON-out. Two call styles:
//!
//! * [`ops`] — the registered `tensor.*` op names ([`TENSOR_OPS`]).
//! * [`call`] — `call("tensor.add", args_json) -> result_json`,
//!   dispatching through [`tensor_op_registry`] exactly as a `.hnl`
//!   `call_host("tensor.add")` would.
//! * [`resolve`]/[`signature`]/[`call_handle`] — the `ffi.rust.resolve`
//!   path (P-013/T-152): the atom name resolves through the shared
//!   `handl-kit-core` module table to an [`AtomHandle`], the external-
//!   caller counterpart of the `{fn, sig}` arrow the in-eval
//!   `ffi.rust.resolve` op mints. `resolve` returns an opaque handle id;
//!   `call_handle` invokes it with the same JSON codec.
//!
//! Args/results are `serde_json::Value` on both sides: tensors ride the
//! `{"__tensor__": {"shape": [...], "data": [...], "dtype": "f64"}}`
//! wire record (see `src/ops/codec.rs`). JSON has no NaN/Inf literals,
//! so non-finite payload elements cross as `{"__nonfinite__": "nan" |
//! "inf" | "-inf"}` objects inside `data` — decoded exactly by
//! [`crate::ops::codec`]'s tensor codec.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use handl_kit_ffi::AtomHandle;
use handl_platform::{EvalContext, EvalError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;

use crate::ops::{TENSOR_OPS, tensor_op_registry};

/// The process-wide op registry — built once, shared by every call.
fn registry() -> &'static handl_plugin_lib_handl::OpRegistry {
    static REGISTRY: OnceLock<handl_plugin_lib_handl::OpRegistry> = OnceLock::new();
    REGISTRY.get_or_init(tensor_op_registry)
}

// ---------------------------------------------------------------------------
// ffi.rust.resolve handles (P-013/T-152)
// ---------------------------------------------------------------------------
//
// `call` below dispatches through the crate-local `tensor_op_registry`.
// `resolve`/`call_handle`/`signature` are the *ffi.rust.resolve* path:
// the atom name resolves through the shared `handl-kit-core` module
// table (the same table the `.hnl` `ffi.rust.resolve` op consults) to
// an owned [`AtomHandle`], and calls go through
// `Module::function(local)` — i.e. the kit-registered `tensor` module's
// op table, not this crate's private registry. External callers can't
// apply the `{fn, sig}` arrow the in-eval op mints, so the handle IS
// the callable: `resolve` returns an opaque id, `call_handle` invokes
// it, args/results crossing as the same JSON wire codec `call` uses.
//
// Handles are process-lifetime (like `handl-plugin-ffi`'s symbol
// caches): atoms are cheap `Arc`s and the table never evicts.

/// Resolved atom handles minted by [`resolve`], keyed by opaque id.
static HANDLES: OnceLock<Mutex<HashMap<u64, AtomHandle>>> = OnceLock::new();
/// Monotonic handle-id source.
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

fn handles() -> &'static Mutex<HashMap<u64, AtomHandle>> {
    HANDLES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Lock a cache mutex, recovering the guard when a previous holder
/// panicked — same poison-tolerance as `handl-plugin-ffi`'s caches.
fn lock_handles() -> std::sync::MutexGuard<'static, HashMap<u64, AtomHandle>> {
    handles().lock().unwrap_or_else(|e| e.into_inner())
}

/// List the registered `tensor.*` op names (declaration order).
#[pyfunction]
fn ops() -> Vec<&'static str> {
    TENSOR_OPS.to_vec()
}

/// Resolve a qualified atom name (`"tensor.add"`) through the kit module
/// table — the PyO3 face of `ffi.rust.resolve`.
///
/// Returns an opaque handle id for [`call_handle`]/[`signature`]. The
/// name may be any atom in the process-wide module table, not just
/// `tensor.*` — whatever kits the host process registered.
///
/// Raises `ValueError` on an unknown module/function or a malformed
/// (unqualified) name — the same gate `ffi.rust.resolve` applies.
#[pyfunction]
fn resolve(op_name: &str) -> PyResult<u64> {
    let handle = crate::ffi::resolve(op_name).map_err(eval_err_to_py)?;
    let id = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    lock_handles().insert(id, handle);
    Ok(id)
}

/// The declared signature (T-148 type value, JSON-encoded) of a handle
/// minted by [`resolve`], or `None` for unsigned atoms.
#[pyfunction]
fn signature(handle: u64) -> PyResult<Option<String>> {
    let map = lock_handles();
    let h = map.get(&handle).ok_or_else(|| {
        PyValueError::new_err(format!("handl_host_tensor.signature: no handle {handle}"))
    })?;
    h.signature()
        .map(|sig| {
            serde_json::to_string(sig).map_err(|e| {
                PyValueError::new_err(format!("handl_host_tensor.signature: encode: {e}"))
            })
        })
        .transpose()
}

/// Call a handle minted by [`resolve`] with a JSON args array —
/// `call_handle(id, "[1.0, 2.0]") -> "<result json>"`.
///
/// Same codec and error contract as [`call`]: the result is a JSON
/// string (wire tensor, array of them, or scalar); `EvalError`s become
/// `ValueError`.
#[pyfunction]
fn call_handle(handle: u64, args_json: &str) -> PyResult<String> {
    let args: Vec<Value> = serde_json::from_str(args_json).map_err(|e| {
        PyValueError::new_err(format!("handl_host_tensor.call_handle: bad args_json: {e}"))
    })?;
    // Clone the handle out and drop the lock before invoking — a called
    // atom must never observe (or deadlock on) the handle-table lock.
    let h = {
        let map = lock_handles();
        map.get(&handle).cloned()
    }
    .ok_or_else(|| {
        PyValueError::new_err(format!("handl_host_tensor.call_handle: no handle {handle}"))
    })?;
    let out = crate::ffi::call(&h, &args).map_err(eval_err_to_py)?;
    serde_json::to_string(&out).map_err(|e| {
        PyValueError::new_err(format!("handl_host_tensor.call_handle: result encode: {e}"))
    })
}

/// Call one `tensor.*` op.
///
/// `op_name` is the fully-qualified name (`"tensor.add"`); `args_json`
/// is a JSON array of positional arguments (wire tensors, numbers, or
/// int lists — whatever the op's ATen port decodes).  Returns the
/// result serialized as a JSON string: a wire tensor object, a JSON
/// array of them (multi-output ops like `unbind`/`chunk`), or a scalar.
///
/// Raises `ValueError` on an unknown op, malformed `args_json`, or any
/// `EvalError` the kernel returns (arity/type/range violations).
#[pyfunction]
fn call(op_name: &str, args_json: &str) -> PyResult<String> {
    let args: Vec<Value> = serde_json::from_str(args_json).map_err(|e| {
        PyValueError::new_err(format!("handl_host_tensor.call: bad args_json: {e}"))
    })?;
    let func = registry().get(op_name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "handl_host_tensor.call: no op registered as {op_name:?}"
        ))
    })?;
    let mut ctx = EvalContext::new();
    let out = func.call(&args, &mut ctx).map_err(eval_err_to_py)?;
    serde_json::to_string(&out)
        .map_err(|e| PyValueError::new_err(format!("handl_host_tensor.call: result encode: {e}")))
}

fn eval_err_to_py(e: EvalError) -> PyErr {
    PyValueError::new_err(format!("handl_host_tensor: {e}"))
}

/// The `handl_host_tensor` extension module.
#[pymodule]
fn handl_host_tensor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ops, m)?)?;
    m.add_function(wrap_pyfunction!(call, m)?)?;
    m.add_function(wrap_pyfunction!(resolve, m)?)?;
    m.add_function(wrap_pyfunction!(signature, m)?)?;
    m.add_function(wrap_pyfunction!(call_handle, m)?)?;
    Ok(())
}
