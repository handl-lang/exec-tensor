//! Opt-in `f32` compute kernels — the `HOST_TENSOR_F32` gate.
//!
//! The tensor repr stays dense row-major `f64` end to end (the wire
//! codec, every op boundary, and [`Tensor`](super::codec::Tensor)
//! itself are unchanged).  When `HOST_TENSOR_F32` is set,
//! *compute-bound* kernels — the matmul family, `softmax`,
//! `layer_norm`, `rms_norm` — down-convert their inputs to `f32`, run
//! the kernel in `f32`, and widen the result back (`f32 → f64` is
//! exact, so the output `Tensor` keeps the same repr).  This halves
//! the working-set bytes inside the hot loops and doubles the SIMD
//! lane width vs `f64` — the ~2-4× the dense-f64 runtime is missing
//! against torch eager.
//!
//! Memory-bound kernels deliberately keep the `f64` path: elementwise
//! broadcast and the generic reduction odometer are bandwidth-bound,
//! and the extra `f64 → f32 → f64` convert passes cost more traffic
//! than `f32` arithmetic saves (measured — see the perf report).
//!
//! Gate semantics mirror the `HOST_TENSOR_CACHE` load cache: unset,
//! empty, or `0`/`false`/`no`/`off` (case-insensitive) → `f64`
//! everywhere, bit-identical to before; anything else opts in.  `f64`
//! remains the default dtype — `f32` is never inferred from the wire.

/// Env var opting into `f32` compute kernels.
const F32_ENV: &str = "HOST_TENSOR_F32";

/// Whether `HOST_TENSOR_F32` opts in — read per kernel entry (env
/// lookup is ~ns against ms-scale kernels; tests can flip it between
/// calls).
pub(crate) fn enabled() -> bool {
    match std::env::var(F32_ENV) {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        ),
        Err(_) => false,
    }
}

/// `&[f64]` → `Vec<f32>` — round-to-nearest on ingest.  For
/// f32-sourced values (safetensors `F32`, `__tensor_bin__` `"f32"`)
/// the stored `f64`s are already f32-representable, so the cast is
/// exact; for f64-native inputs the cast *is* the f32 semantic
/// (torch would have stored the rounded value).
pub(crate) fn to_f32(data: &[f64]) -> Vec<f32> {
    data.iter().map(|&x| x as f32).collect()
}

/// `Vec<f32>` → `Vec<f64>` — exact widening (`f32 → f64` is lossless).
pub(crate) fn widen(data: Vec<f32>) -> Vec<f64> {
    data.into_iter().map(f64::from).collect()
}
