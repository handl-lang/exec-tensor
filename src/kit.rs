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
pub const MODULES: &[(&str, ModuleCtor)] = &[(MODULE_NAME, |_: &KitResolver, e| {
    ModuleEntry::Handl(build_tensor_module().with_eval_engine(e))
})];

/// Register the `tensor` module into the shared kit table. `Once`-safe.
pub fn register() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| handl_kit_core::register_modules(MODULES));
}

// ---------------------------------------------------------------------------
// Atom-provider self-registration (P-013 phase-2a, proposal-0012)
// ---------------------------------------------------------------------------
//
// This crate submits one `AtomProvider` so dep binding
// (`handl_plugin_lib_handl::dep_atom_modules`) can pair a dep's
// `atoms.yaml` with `tensor_impls()` — no hand table row, no host-side
// `register()` call. The inventory only sees submissions from crates
// that are *linked*: any consumer must add `handl-host-tensor` as a
// Cargo dependency AND reference one of its symbols, or the linker
// drops this object and the provider is invisible.
//
// [`atom_provider_link_anchor`] is that reference — placed in this file
// next to `submit!` so the archive member carrying the registration is
// the one referenced. `handl-kit-prelude`'s `register()` calls it;
// hosts composing a kit without prelude must call it themselves to get
// dep-bound `tensor` atoms.

/// The `atoms.yaml` spec this crate's impl table serves.
pub const ATOMS_YAML: &str = include_str!("../atoms.yaml");

inventory::submit! {
    handl_plugin_lib_handl::atoms::AtomProvider {
        modules: &[MODULE_NAME],
        atoms_yaml: ATOMS_YAML,
        host_impls: Some(crate::ops::tensor_impls),
        kernel_impls: None,
    }
}

/// Link anchor for the `AtomProvider` submission above — see the
/// comment block. Referencing this fn pulls the object containing
/// `inventory::submit!` (and, through its `tensor_impls` reference,
/// the impl table) into the final artifact.
#[doc(hidden)]
pub fn atom_provider_link_anchor() {}
