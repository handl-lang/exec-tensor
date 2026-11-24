//! Kit registration — expose the `tensor` module through
//! `handl-kit-core`'s process-wide module table so hosts that assemble a
//! [`handl_kit_core::KitResolver`]-backed kit get `tensor.*` host ops
//! without wiring the [`HandlModule`] themselves.
//!
//! This is the same composition `mlopt-kit`'s `CatoptKit` performs inline:
//! `HandlModule::new("tensor", TENSOR_HNL, &[]).with_ops(TENSOR_OPS,
//! tensor_op_registry())`. Any host that calls [`register`] —
//! `handl-kit-prelude`'s canonical kit (so `hnl run`/`hnl eval` execute
//! `tensor.*` natively), `hnl-ext-tensor`'s self-contained runner, or a
//! consumer's own `Kit` assembly — resolves `tensor.<op>` calls through
//! the shared table.
//!
//! [`register`] is `Once`-safe: repeat calls are no-ops.

use std::sync::Arc;

use handl_kit_core::{KitResolver, ModuleCtor, ModuleEntry};
use handl_plugin_lib_handl::HandlModule;

use crate::{TENSOR_HNL, TENSOR_OPS, tensor_op_registry};

/// The module name this registration provides (`"tensor"`).
pub const MODULE_NAME: &str = "tensor";

/// Build the `tensor` [`HandlModule`]: the authored `tensor.hnl` surface
/// with every op in [`TENSOR_OPS`] attached to the native
/// [`tensor_op_registry`] (signatures included — `register_with_sig`
/// happens inside the registry builder, T-148).
fn build_tensor_module() -> HandlModule {
    HandlModule::new(MODULE_NAME, TENSOR_HNL, &[])
        .with_ops(TENSOR_OPS, Arc::new(tensor_op_registry()))
}

/// The kit's module table — just `tensor`.
pub const MODULES: &[(&'static str, ModuleCtor)] = &[(MODULE_NAME, |_: &KitResolver, e| {
    ModuleEntry::Handl(build_tensor_module().with_eval_engine(e))
})];

/// Register the `tensor` module into the shared kit table. `Once`-safe.
pub fn register() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| handl_kit_core::register_modules(MODULES));
}
