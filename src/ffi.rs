//! `ffi.rust.resolve` atom handles for external callers (P-013/T-152).
//!
//! Thin wrapper over [`handl_kit_ffi::AtomHandle`]: ensures this crate's
//! `tensor` module is registered in the shared `handl-kit-core` module
//! table ([`crate::kit::register`] — `Once`-safe), then resolves
//! qualified atom names (`"tensor.add"`) to owned callable handles. The
//! PyO3 binding ([`crate::python`], `--features python`) exposes
//! `resolve`/`call_handle`/`signature` on top; Rust consumers can call
//! [`resolve`] directly.
//!
//! This is the runtime-caller half of the `ffi.rust.resolve` contract:
//! inside `.hnl` eval the op mints a `{fn, sig}` wrapper arrow that
//! re-dispatches through the applying runtime's module chain; outside
//! eval there is no arrow to apply, so the handle *is* the callable —
//! args/results cross as `serde_json::Value` through
//! [`Function::call`]'s codec, the same wire format `tensor.*` ops
//! already use (`{"__tensor__": …}` records).

use handl_kit_ffi::AtomHandle;
use handl_platform::{EvalContext, EvalError};

/// Resolve a qualified atom name (`"mod.fn"`) through the kit module
/// table — the same existence + signature gate `ffi.rust.resolve`
/// applies inside `.hnl` eval.
///
/// [`crate::kit::register`] runs first so `tensor.*` resolves even in
/// processes that never assembled a kit (the PyO3 cdylib being the
/// canonical case).
pub fn resolve(name: &str) -> Result<AtomHandle, EvalError> {
    crate::kit::register();
    AtomHandle::resolve(name)
}

/// Call a resolved handle with already-evaluated JSON arguments.
///
/// Equivalent to [`AtomHandle::call`] with a fresh [`EvalContext`] — the
/// shape external callers (PyO3, wasm, C-ABI shims) need, since they
/// have no ambient eval context to lend.
pub fn call(
    handle: &AtomHandle,
    args: &[serde_json::Value],
) -> Result<serde_json::Value, EvalError> {
    let mut ctx = EvalContext::new();
    handle.call(args, &mut ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The consumer proof (P-013/T-152): `tensor.add` resolved through
    /// `ffi.rust.resolve`'s handle path — not the crate-local
    /// `tensor_op_registry` the PyO3 `call` op uses — returns the same
    /// `__tensor__` wire record.
    #[test]
    fn tensor_add_resolves_and_calls_through_kit_table() {
        let handle = resolve("tensor.add").expect("tensor.add resolves");
        assert_eq!(handle.name(), "tensor.add");
        // `tensor.add` declares a sig in atoms.yaml → the handle
        // carries the T-148 type value (or None if unsigned — either
        // way the call below is the proof).
        let _ = handle.signature();

        let a = json!({"__tensor__": {"shape": [2], "data": [1.0, 2.0], "dtype": "f64"}});
        let b = json!({"__tensor__": {"shape": [2], "data": [10.0, 20.0], "dtype": "f64"}});
        let out = call(&handle, &[a, b]).expect("tensor.add call");
        assert_eq!(out["__tensor__"]["data"], json!([11.0, 22.0]));
    }

    #[test]
    fn unknown_atoms_are_rejected_at_resolve() {
        assert!(resolve("tensor.no_such_op").is_err());
        assert!(resolve("no.such").is_err());
        assert!(resolve("bare").is_err());
    }
}
