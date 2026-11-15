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
//! `Function` contract is already JSON-in/JSON-out, so the binding is
//! two functions:
//!
//! * [`ops`] — the registered `tensor.*` op names ([`TENSOR_OPS`]).
//! * [`call`] — `call("tensor.add", args_json) -> result_json`,
//!   dispatching through [`tensor_op_registry`] exactly as a `.hnl`
//!   `call_host("tensor.add")` would.
//!
//! Args/results are `serde_json::Value` on both sides: tensors ride the
//! `{"__tensor__": {"shape": [...], "data": [...], "dtype": "f64"}}`
//! wire record (see `src/ops/codec.rs`). JSON has no NaN/Inf literals,
//! so non-finite payload elements cross as `{"__nonfinite__": "nan" |
//! "inf" | "-inf"}` objects inside `data` — decoded exactly by
//! [`crate::ops::codec`]'s tensor codec.

use std::sync::OnceLock;

use handl_platform::{EvalContext, EvalError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;

use crate::ops::{tensor_op_registry, TENSOR_OPS};

/// The process-wide op registry — built once, shared by every call.
fn registry() -> &'static handl_plugin_lib_handl::OpRegistry {
    static REGISTRY: OnceLock<handl_plugin_lib_handl::OpRegistry> = OnceLock::new();
    REGISTRY.get_or_init(tensor_op_registry)
}

/// List the registered `tensor.*` op names (declaration order).
#[pyfunction]
fn ops() -> Vec<&'static str> {
    TENSOR_OPS.to_vec()
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
    Ok(())
}
