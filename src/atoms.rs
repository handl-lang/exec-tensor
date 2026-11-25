//! # `atoms.yaml` — spec-driven op registry (P-013/T-149)
//!
//! The atom IDL replaces hand-wired `*_op_registry()` builders: a package
//! ships an `atoms.yaml` spec declaring each atom's `{name, sig, impl}`
//! and the loader builds [`OpRegistry`] entries `{name → {fn, sig}}`
//! from it — the op names, signatures, and fn bindings all come from the
//! spec plus the impl table.
//!
//! ## Spec format
//!
//! One file per package (`atoms.yaml` at the repo root, vendored by
//! `handl-pkg-vendor` like `pkg.hnl`):
//!
//! ```yaml
//! module: tensor
//! atoms:
//!   - name: tensor.zeros
//!     sig: "(List Int) ~> Tensor"
//!     impl: rust_tensor_zeros
//!     total: true
//! ```
//!
//! Fields per atom (schema identical to `handl-plugin-lib-stdlib`'s
//! `atoms.rs`, the P-013/T-149 reference implementation):
//!
//! - `name` — the qualified dispatch key (`mod.fn`).
//! - `sig` — handl signature text (`~>` canonical), interpreted by
//!   [`sig_to_type_value`] into the tagged checker type value stored on
//!   the registry entry (T-148). Optional: absent = unsigned atom.
//! - `impl` — key into the module's impl table (`impl:` in yaml —
//!   `impl` is a Rust keyword, hence the rename): a
//!   `(&'static str, Arc<dyn Function + Send + Sync>)` row.
//! - `total` — `guarantee: True` decl clause marker.
//! - `caps` — capability names the atom *requires* of the consuming
//!   runtime (T-151 caps-as-data): a `pkg.hnl` dep scope's `caps:`
//!   allowlist filters atoms whose requirements it does not cover.
//!   Absent/empty = no requirements.
//! - `requires` / `termination` / `guarantee` — the remaining index.hnl
//!   decl-clause texts carried verbatim (`guarantee:` excludes `True`,
//!   which stays `total`). Carried as data, not yet consumed.
//!
//! ## Provenance — local port, upstream move pending
//!
//! This module is a **line-for-line port** of
//! `handl-plugin-lib-stdlib/src/atoms.rs` (host repo, P-013/T-149),
//! re-exported for crates that cannot depend on the stdlib plugin.
//! lib-tensor-native's dep closure (`handl-platform` +
//! `handl-plugin-lib-handl`) already carries everything the loader
//! needs — [`OpRegistry`], [`sig_to_type_value`] — but the yaml
//! machinery itself lives one crate too high. The correct upstream
//! home is `handl-plugin-lib-handl` (it owns `OpRegistry` and
//! `atom_sig`); that move was deliberately **not** done here because
//! host's `plugin-lib-handl` has in-flight uncommitted P-013/T-152 work.
//! Follow-up: move `atoms.rs` to `handl-plugin-lib-handl`, re-export it
//! from `handl-plugin-lib-stdlib`, and delete this port.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use handl_platform::Function;
use handl_plugin_lib_handl::{OpRegistry, sig_to_type_value};
use serde::Deserialize;

/// One `impl:` table entry: spec key → the `Function` impl (the handl).
pub type AtomImpl = (&'static str, Arc<dyn Function + Send + Sync>);

/// A module's impl table — what `ops::<mod>::*_impls()` returns.
pub type ImplTable = fn() -> Vec<AtomImpl>;

#[derive(Debug, Deserialize)]
struct RawSpec {
    /// Single-module form: `module: tensor` + top-level `atoms:`.
    module: Option<String>,
    #[serde(default)]
    atoms: Vec<AtomDecl>,
    /// Multi-module form: `modules: { math: { atoms: [...] } }`.
    #[serde(default)]
    modules: BTreeMap<String, RawModule>,
}

#[derive(Debug, Deserialize)]
struct RawModule {
    #[serde(default)]
    atoms: Vec<AtomDecl>,
}

/// One atom declaration — a row of the IDL.
#[derive(Debug, Clone, Deserialize)]
pub struct AtomDecl {
    /// Qualified dispatch key, e.g. `"tensor.zeros"`.
    pub name: String,
    /// Handl signature text (`"(List Int) ~> Tensor"`); `None` =
    /// unsigned atom (dynamic-top typecheck behavior).
    #[serde(default)]
    pub sig: Option<String>,
    /// Key into the impl table (`impl:` in yaml — `impl` is a Rust
    /// keyword, hence the rename).
    #[serde(rename = "impl")]
    pub impl_key: String,
    /// `guarantee: True` decl clause.
    #[serde(default)]
    pub total: bool,
    /// Capability names this atom requires of the consuming runtime
    /// (T-151 caps-as-data). See the module docs for consumers.
    #[serde(default)]
    pub caps: Vec<String>,
    /// `requires:` decl-clause texts, carried verbatim.
    #[serde(default)]
    pub requires: Vec<String>,
    /// `termination:` decl-clause texts, carried verbatim.
    #[serde(default)]
    pub termination: Vec<String>,
    /// Non-`True` `guarantee:` decl-clause texts (`True` is `total`).
    #[serde(default)]
    pub guarantee: Vec<String>,
}

/// Union of every atom's `caps` requirements (T-151) — kept in parity
/// with the upstream `handl-plugin-lib-handl` port.
#[allow(dead_code)]
pub fn atom_capabilities(atoms: &[AtomDecl]) -> BTreeSet<String> {
    atoms.iter().flat_map(|a| a.caps.iter().cloned()).collect()
}

/// Parse an `atoms.yaml` document into `(module_name, atoms)` pairs.
///
/// Accepts both the single-module (`module:` + `atoms:`) and the
/// multi-module (`modules:` map) shapes; the single-module form yields
/// exactly one pair. Returns an error on malformed yaml or a document
/// declaring neither shape.
pub fn parse_atoms(yaml_text: &str) -> Result<Vec<(String, Vec<AtomDecl>)>, String> {
    let raw: RawSpec =
        serde_yaml::from_str(yaml_text).map_err(|e| format!("atoms.yaml parse: {e}"))?;
    if !raw.modules.is_empty() {
        return Ok(raw
            .modules
            .into_iter()
            .map(|(name, m)| (name, m.atoms))
            .collect());
    }
    match raw.module {
        Some(module) => Ok(vec![(module, raw.atoms)]),
        None => Err("atoms.yaml: neither `module:`+`atoms:` nor `modules:` present".to_string()),
    }
}

/// Select one module's atom declarations from a parsed spec document.
///
/// `module` is the package module name (e.g. `"tensor"`). For a
/// single-module document the declared `module:` must match; for a
/// `modules:` map the key must exist.
pub fn module_atoms(yaml_text: &str, module: &str) -> Result<Vec<AtomDecl>, String> {
    let specs = parse_atoms(yaml_text)?;
    let known: Vec<String> = specs.iter().map(|(n, _)| n.clone()).collect();
    specs
        .into_iter()
        .find(|(name, _)| name == module)
        .map(|(_, atoms)| atoms)
        .ok_or_else(|| format!("atoms.yaml: no module `{module}` (declared: {known:?})"))
}

/// Build an [`OpRegistry`] from atom declarations + an impl table.
///
/// Each atom's `impl:` key resolves to a [`Function`] impl; the atom's
/// `sig` (when present and parseable) is attached via
/// [`OpRegistry::register_with_sig`] so typecheck sees the declared
/// signature (T-148). Errors:
///
/// - `impl:` key missing from the table → error (spec/impl drift).
/// - `sig` present but unparseable → error (a spec bug must not
///   silently degrade to the dynamic top).
/// - duplicate `name` → last one wins (same replace semantics as
///   [`OpRegistry::register`]).
pub fn registry_from_atoms(atoms: &[AtomDecl], impls: &[AtomImpl]) -> Result<OpRegistry, String> {
    let mut registry = OpRegistry::new();
    for atom in atoms {
        let func = impls
            .iter()
            .find(|(key, _)| *key == atom.impl_key)
            .map(|(_, f)| Arc::clone(f))
            .ok_or_else(|| {
                format!(
                    "atoms.yaml: atom `{}`: no impl `{}` in table",
                    atom.name, atom.impl_key
                )
            })?;
        match &atom.sig {
            Some(text) => {
                let sig = sig_to_type_value(text).ok_or_else(|| {
                    format!("atoms.yaml: atom `{}`: unparseable sig `{text}`", atom.name)
                })?;
                registry.register_with_sig(&atom.name, func, sig);
            }
            None => registry.register(&atom.name, func),
        }
    }
    Ok(registry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Echo;
    impl Function for Echo {
        fn call(
            &self,
            _args: &[serde_json::Value],
            _ctx: &mut handl_platform::EvalContext,
        ) -> Result<serde_json::Value, handl_platform::EvalError> {
            Ok(json!(null))
        }
    }

    const SPEC: &str = r#"
module: tensor
atoms:
  - name: tensor.zeros
    sig: "(List Int) ~> Tensor"
    impl: rust_tensor_zeros
    total: true
  - name: tensor.load
    sig: "Value... ~> Tensor"
    impl: rust_tensor_load
"#;

    fn impls() -> Vec<AtomImpl> {
        vec![
            ("rust_tensor_zeros", Arc::new(Echo)),
            ("rust_tensor_load", Arc::new(Echo)),
        ]
    }

    #[test]
    fn parses_single_module_spec() {
        let atoms = module_atoms(SPEC, "tensor").unwrap();
        assert_eq!(atoms.len(), 2);
        assert_eq!(atoms[0].name, "tensor.zeros");
        assert_eq!(atoms[0].impl_key, "rust_tensor_zeros");
        assert!(atoms[0].total);
        assert_eq!(atoms[1].sig.as_deref(), Some("Value... ~> Tensor"));
    }

    #[test]
    fn registry_from_atoms_carries_fn_and_sig() {
        let atoms = module_atoms(SPEC, "tensor").unwrap();
        let registry = registry_from_atoms(&atoms, &impls()).unwrap();
        assert_eq!(registry.len(), 2);
        assert!(registry.contains("tensor.zeros"));
        // Sig attached → tagged type.function value.
        let sig = registry.signature("tensor.zeros").unwrap();
        assert_eq!(sig["__tag__"], "type.function");
        assert_eq!(sig["body"]["__tag__"], "type.unknown"); // Tensor → top
    }

    #[test]
    fn missing_impl_and_bad_sig_are_errors() {
        let atoms = module_atoms(SPEC, "tensor").unwrap();
        assert!(registry_from_atoms(&atoms, &[]).is_err());
        let bad = vec![AtomDecl {
            name: "tensor.x".into(),
            sig: Some("not a ~> sig !!!".into()),
            impl_key: "rust_tensor_zeros".into(),
            total: false,
            caps: vec![],
            requires: vec![],
            termination: vec![],
            guarantee: vec![],
        }];
        assert!(registry_from_atoms(&bad, &impls()).is_err());
    }
}
