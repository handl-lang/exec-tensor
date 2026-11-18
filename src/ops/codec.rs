//! The `__tensor__` wire codec and the boundary ops — the impl mirrored by
//! lib-tensor's `codec.hnl`.
//!
//! [`Tensor`] is the decoded representation (dense, row-major, `f64`);
//! [`decode_tensor`] / [`encode_tensor`] cross the `serde_json::Value`
//! `Function` boundary in the `{"__tensor__": {"shape": ..., "data": ...,
//! "dtype": "f64"}}` tagged-object style (ADR-0083 §2.1). The `require_*`
//! / `opt_*` helpers decode positional arguments for every op — including
//! the optional trailing parameters of `conv2d` / `max_pool` /
//! `batch_norm`, where a `null` or absent argument selects the default.
//!
//! The boundary ops live here too: [`Zeros`], [`Ones`], [`FromList`]
//! construct wire tensors; [`ToList`], [`Shape`] project them.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

use handl_platform::{EvalContext, EvalError, Function};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// The tensor codec
// ---------------------------------------------------------------------------

/// A decoded tensor: dense, row-major, `f64`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Tensor {
    /// Extent per axis; `shape.len()` is the rank.
    pub(crate) shape: Vec<usize>,
    /// Flat row-major payload; `data.len() == product(shape)`.
    pub(crate) data: Vec<f64>,
}

pub(crate) fn type_mismatch(expected: &str, got: &str) -> EvalError {
    EvalError::TypeMismatch(format!("expected {expected}, got {got}"))
}

/// The non-finite wire tag — `serde_json` has no NaN/±inf literals, so a
/// non-finite `f64` element serializes as the tagged object
/// `{"__nonfinite__": "nan" | "inf" | "-inf"}` inside `data`, the same
/// `__x__` tagged-object convention as `__tensor__`/`__bytes__`.
const NONFINITE_TAG: &str = "__nonfinite__";

/// Decode one `data` element: a JSON number, or the
/// `{"__nonfinite__": …}` tag for NaN/`inf`/`-inf`. `None` on anything
/// else (including a `null`, an unknown tag, or a tag object nested
/// where a number belongs).
fn decode_f64(x: &Value) -> Option<f64> {
    if let Some(n) = x.as_f64() {
        return Some(n);
    }
    match x.get(NONFINITE_TAG)?.as_str()? {
        "nan" => Some(f64::NAN),
        "inf" => Some(f64::INFINITY),
        "-inf" => Some(f64::NEG_INFINITY),
        _ => None,
    }
}

/// Encode one element for the wire: finite `f64`s stay JSON numbers;
/// NaN/`inf`/`-inf` become the `{"__nonfinite__": …}` tag (there is no
/// JSON literal for them — `Number::from_f64` rejects them and they
/// would otherwise serialize as `null`, which decode rejects).
fn encode_f64(x: f64) -> Value {
    if x.is_nan() {
        json!({ NONFINITE_TAG: "nan" })
    } else if x == f64::INFINITY {
        json!({ NONFINITE_TAG: "inf" })
    } else if x == f64::NEG_INFINITY {
        json!({ NONFINITE_TAG: "-inf" })
    } else {
        json!(x)
    }
}

/// Decode a tensor argument. Accepts `{"__tensor__": {"shape": [...],
/// "data": [...], "dtype"?: "f64"}}` and, since ADR-0179, the binary
/// spelling `"data": {"__tensor_bin__": "<base64 LE>"}` with a required
/// `"dtype": "f32"|"f64"` — the payload widens `f32 → f64` exactly, so
/// [`Tensor`] stays `Vec<f64>` either way. Anything else — a missing
/// tag, a non-array shape, a negative or non-integer extent, a data
/// element that is neither a number nor a valid `{"__nonfinite__": …}`
/// tag, a bad base64/length `__tensor_bin__` payload, a `dtype` that
/// does not match the `data` spelling (`"f64"`/absent for the array
/// form, `"f32"`/`"f64"` for the binary form), or a data length that
/// is not `product(shape)` — is a `TypeMismatch` (malformed wire
/// value) rather than a silent coercion.
pub(crate) fn decode_tensor(arg: &Value) -> Result<Tensor, EvalError> {
    let inner = arg
        .get("__tensor__")
        .and_then(Value::as_object)
        .ok_or_else(|| type_mismatch("Tensor", "non-Tensor"))?;
    let dtype = match inner.get("dtype") {
        None => None,
        Some(d) => Some(
            d.as_str()
                .ok_or_else(|| type_mismatch("dtype \"f64\"", "other dtype"))?,
        ),
    };
    let shape_val = inner
        .get("shape")
        .ok_or_else(|| type_mismatch("tensor.shape field", "missing"))?;
    let shape_arr = shape_val
        .as_array()
        .ok_or_else(|| type_mismatch("shape: array", "non-array"))?;
    let mut shape = Vec::with_capacity(shape_arr.len());
    for d in shape_arr {
        let n = d
            .as_u64()
            .ok_or_else(|| type_mismatch("shape: non-negative integers", "non-integer"))?;
        shape.push(
            usize::try_from(n).map_err(|_| type_mismatch("shape: usize extents", "overflow"))?,
        );
    }
    let data_val = inner
        .get("data")
        .ok_or_else(|| type_mismatch("tensor.data field", "missing"))?;
    let data: Vec<f64> = if data_val.is_object() {
        // Binary spelling (ADR-0179): `{"__tensor_bin__": "<base64 LE>"}`
        // — `dtype` is required and must be "f32" or "f64"; the decoded
        // payload widens to the f64 `Tensor` (f32→f64 is exact).
        let dt = dtype
            .ok_or_else(|| type_mismatch("dtype \"f32\"|\"f64\" for __tensor_bin__", "missing"))?;
        if dt != "f32" && dt != "f64" {
            return Err(type_mismatch(
                "dtype \"f32\"|\"f64\" for __tensor_bin__",
                "other dtype",
            ));
        }
        handl_value::tensor_bin_decode(&shape, dt, data_val)
            .map(|td| td.to_f64_vec())
            .ok_or_else(|| type_mismatch("__tensor_bin__ payload", "malformed"))?
    } else {
        // Array spelling: `dtype` is optional but only `"f64"` is legal —
        // the array carries widened f64 elements (an "f32" array is
        // malformed).
        if dtype.is_some_and(|d| d != "f64") {
            return Err(type_mismatch("dtype \"f64\"", "other dtype"));
        }
        let data_arr = data_val
            .as_array()
            .ok_or_else(|| type_mismatch("data: array", "non-array"))?;
        let mut data = Vec::with_capacity(data_arr.len());
        for x in data_arr {
            data.push(decode_f64(x).ok_or_else(|| {
                type_mismatch("data: numbers or __nonfinite__ tags", "non-number")
            })?);
        }
        data
    };
    let expected = shape.iter().product::<usize>();
    if data.len() != expected {
        return Err(EvalError::Other(format!(
            "tensor: data length {} does not match product of shape {shape:?} ({expected})",
            data.len()
        )));
    }
    Ok(Tensor { shape, data })
}

/// Decode the tensor at argument position `pos`.
pub(crate) fn require_tensor(args: &[Value], pos: usize) -> Result<Tensor, EvalError> {
    let arg = args
        .get(pos)
        .ok_or_else(|| type_mismatch("tensor argument", "missing"))?;
    decode_tensor(arg)
}

/// Encode a tensor as the canonical `{"__tensor__": {...}}` wire value.
///
/// Non-finite `f64` elements (`inf`, `-inf`, `NaN` — e.g. a `div` by
/// zero or an `add`/`mul` overflow) have no `serde_json` number
/// representation, so they serialize as `{"__nonfinite__": "nan" |
/// "inf" | "-inf"}` elements of `data` — round-tripping exactly through
/// [`decode_tensor`]. Finite tensors encode byte-identically to the
/// pre-tag wire form, so old payloads still decode.
///
/// Note the boundary asymmetry that remains: the kernel
/// `TaggedValue::Tensor` carries non-finite `f64`s inside an engine,
/// but other `Value` egress points must call
/// `handl_value::materialize_tensor_bin` before printing or they emit
/// the ADR-0179 binary spelling (see `hnl-run`).
///
/// ADR-0179: tensors with `numel >= TENSOR_BIN_THRESHOLD` encode as
/// the binary `{"__tensor_bin__": "<base64 LE f64>"}` spelling with
/// `dtype: "f64"` — [`decode_tensor`] reads both forms, so the wire is
/// symmetric. Below the threshold the encoding is byte-identical to
/// the pre-ADR array form.
pub(crate) fn encode_tensor(t: Tensor) -> Value {
    if t.data.len() >= handl_value::TENSOR_BIN_THRESHOLD {
        let payload: Vec<u8> = t.data.iter().flat_map(|x| x.to_le_bytes()).collect();
        // The length invariant makes this infallible; fall through to
        // the array form defensively if it ever isn't.
        if let Some(v) = handl_value::tensor_bin_record(&t.shape, "f64", &payload) {
            return v;
        }
    }
    let data: Vec<Value> = t.data.iter().map(|&x| encode_f64(x)).collect();
    json!({
        "__tensor__": {
            "shape": t.shape,
            "data": data,
            "dtype": "f64",
        }
    })
}

/// Decode a non-negative integer extent (integral floats count — upstream
/// ops produce `f64` values, matching the `require_int` convention in
/// `lib-char-native`).
pub(crate) fn require_extent(arg: &Value) -> Result<usize, EvalError> {
    let n = arg.as_u64().or_else(|| {
        arg.as_f64()
            .filter(|n| *n >= 0.0 && n.fract() == 0.0 && n.is_finite())
            .map(|n| n as u64)
    });
    n.and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| type_mismatch("non-negative integer extent", "other"))
}

/// Decode a `shape` argument (a list of non-negative integer extents).
pub(crate) fn require_shape(args: &[Value], pos: usize) -> Result<Vec<usize>, EvalError> {
    let arg = args
        .get(pos)
        .ok_or_else(|| type_mismatch("shape argument", "missing"))?;
    let arr = arg
        .as_array()
        .ok_or_else(|| type_mismatch("shape: array of extents", "non-array"))?;
    arr.iter().map(require_extent).collect()
}

/// Row-major strides of `shape` (the contiguous layout every tensor in
/// this component uses).
pub(crate) fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }
    strides
}

// ---------------------------------------------------------------------------
// Optional-argument decoding (conv2d / batch_norm)
//
// Optional trailing parameters are positional; a `null` (or an absent
// argument) selects the default, so callers can set a later parameter
// without committing to the earlier ones — `conv2d(x, w, null, [2, 2])`
// is "no bias, stride 2×2, no padding".
// ---------------------------------------------------------------------------

/// Check the argument count is in `min..=max` (variadic ops cannot use
/// `require_arg_count`).
pub(crate) fn require_arg_range(
    args: &[Value],
    min: usize,
    max: usize,
    op: &str,
) -> Result<(), EvalError> {
    if args.len() < min || args.len() > max {
        return Err(EvalError::TypeMismatch(format!(
            "{op}: expected {min}..={max} args, got {}",
            args.len()
        )));
    }
    Ok(())
}

/// Whether the optional argument at `pos` is present and non-null.
pub(crate) fn has_opt_arg(args: &[Value], pos: usize) -> bool {
    args.get(pos).is_some_and(|a| !a.is_null())
}

/// Decode an optional tensor argument (`null`/absent → `None`).
pub(crate) fn opt_tensor(args: &[Value], pos: usize) -> Result<Option<Tensor>, EvalError> {
    match args.get(pos) {
        Some(arg) if !arg.is_null() => Ok(Some(decode_tensor(arg)?)),
        _ => Ok(None),
    }
}

/// Decode an optional spatial parameter (`stride` / `padding`): a single
/// non-negative integer `n` means `[n, n]`; a two-element list `[h, w]`
/// gives the per-axis extents; `null`/absent means `default`.
pub(crate) fn opt_extent_pair(
    args: &[Value],
    pos: usize,
    default: [usize; 2],
    name: &str,
) -> Result<[usize; 2], EvalError> {
    let Some(arg) = args.get(pos) else {
        return Ok(default);
    };
    if arg.is_null() {
        return Ok(default);
    }
    if let Some(arr) = arg.as_array() {
        if arr.len() != 2 {
            return Err(type_mismatch(
                &format!("{name}: int or [h, w] pair"),
                &format!("array of len {}", arr.len()),
            ));
        }
        return Ok([require_extent(&arr[0])?, require_extent(&arr[1])?]);
    }
    let n = require_extent(arg)?;
    Ok([n, n])
}

/// Decode an optional strictly-positive integer (`groups`): `null`/absent
/// means `default`; zero or negative is an error (a non-positive group
/// count has no meaning, unlike a zero stride/padding which is simply
/// rejected downstream or degenerate).
pub(crate) fn opt_positive_int(
    args: &[Value],
    pos: usize,
    default: usize,
    name: &str,
) -> Result<usize, EvalError> {
    if !has_opt_arg(args, pos) {
        return Ok(default);
    }
    let n = require_extent(&args[pos])?;
    if n == 0 {
        return Err(EvalError::Other(format!(
            "{name}: must be a positive integer, got 0"
        )));
    }
    Ok(n)
}

/// Decode an optional non-negative finite float (`eps`): `null`/absent
/// means `default`.
pub(crate) fn opt_nonneg_float(
    args: &[Value],
    pos: usize,
    default: f64,
    name: &str,
) -> Result<f64, EvalError> {
    if !has_opt_arg(args, pos) {
        return Ok(default);
    }
    let x = args[pos]
        .as_f64()
        .ok_or_else(|| type_mismatch(&format!("{name}: number"), "non-number"))?;
    if !x.is_finite() || x < 0.0 {
        return Err(EvalError::Other(format!(
            "{name}: expected a non-negative finite number, got {x}"
        )));
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// The Rust handl for `op "tensor.zeros" : (List Int) -> Tensor`.
#[derive(Debug, Clone, Copy)]
pub struct Zeros;
impl Function for Zeros {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 1)?;
        let shape = require_shape(args, 0)?;
        let len = shape.iter().product();
        Ok(encode_tensor(Tensor {
            shape,
            data: vec![0.0; len],
        }))
    }
}

/// The Rust handl for `op "tensor.ones" : (List Int) -> Tensor`.
#[derive(Debug, Clone, Copy)]
pub struct Ones;
impl Function for Ones {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 1)?;
        let shape = require_shape(args, 0)?;
        let len = shape.iter().product();
        Ok(encode_tensor(Tensor {
            shape,
            data: vec![1.0; len],
        }))
    }
}

/// The Rust handl for `op "tensor.from_list" : Array -> Tensor`.
///
/// Infers the shape from the nesting of number arrays — leaves may also
/// be `{"__nonfinite__": "nan" | "inf" | "-inf"}` tags, the wire spelling
/// of non-finite elements — and requires the input to be rectangular: at
/// every level all siblings must have the same inferred sub-shape, so
/// `[[1, 2], [3]]` and `[[1], 2]` are errors. A scalar argument is not a
/// list — `TypeMismatch`. `[]` is the rank-1 empty tensor (shape `[0]`);
/// `[[], []]` is shape `[2, 0]`.
#[derive(Debug, Clone, Copy)]
pub struct FromList;
impl Function for FromList {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 1)?;
        let arg = &args[0];
        if !arg.is_array() {
            return Err(type_mismatch("nested number arrays", "non-array"));
        }
        let shape = infer_shape(arg)?;
        let mut data = Vec::with_capacity(shape.iter().product());
        flatten(arg, &mut data);
        Ok(encode_tensor(Tensor { shape, data }))
    }
}

/// Infer `xs`' shape: `[]` prepends its length to the (uniform) sub-shape
/// of its elements; a number — or a `{"__nonfinite__": …}` tag, the wire
/// spelling of NaN/±inf — contributes `[]`; anything else errors.
fn infer_shape(xs: &Value) -> Result<Vec<usize>, EvalError> {
    match xs {
        Value::Array(arr) => {
            let mut shape = vec![arr.len()];
            if let Some(first) = arr.first() {
                let sub = infer_shape(first)?;
                for el in &arr[1..] {
                    let el_shape = infer_shape(el)?;
                    if el_shape != sub {
                        return Err(EvalError::Other(format!(
                            "tensor.from_list: ragged input — element shapes {sub:?} and {el_shape:?} differ"
                        )));
                    }
                }
                shape.extend(sub);
            }
            Ok(shape)
        }
        v if decode_f64(v).is_some() => Ok(Vec::new()),
        _ => Err(type_mismatch("nested number arrays", "non-number leaf")),
    }
}

/// Append `xs`' numbers to `out` in row-major order. Only reachable after
/// [`infer_shape`] accepted `xs`, so every leaf decodes via [`decode_f64`]
/// (a JSON number or a `{"__nonfinite__": …}` tag).
fn flatten(xs: &Value, out: &mut Vec<f64>) {
    match xs {
        Value::Array(arr) => {
            for el in arr {
                flatten(el, out);
            }
        }
        leaf => {
            if let Some(x) = decode_f64(leaf) {
                out.push(x);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------

/// The Rust handl for `op "tensor.to_list" : Tensor -> Array`.
///
/// The inverse of `from_list`: re-nests the flat row-major payload into
/// `shape`'s structure. Non-finite elements come back as the
/// `{"__nonfinite__": …}` tag — accepted again by `from_list`, so the
/// pair round-trips NaN/±inf exactly.
#[derive(Debug, Clone, Copy)]
pub struct ToList;
impl Function for ToList {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 1)?;
        let t = require_tensor(args, 0)?;
        let mut it = t.data.iter().copied();
        Ok(nest(&t.shape, &mut it))
    }
}

/// Re-nest `data` (row-major) under `shape`. Non-finite elements emit
/// the `{"__nonfinite__": …}` tag — the same spelling [`encode_tensor`]
/// uses in `data` — so a `to_list` result never silently degrades a
/// NaN/±inf to `null`.
fn nest(shape: &[usize], data: &mut impl Iterator<Item = f64>) -> Value {
    match shape.split_first() {
        None => data.next().map(encode_f64).unwrap_or(Value::Null),
        Some((&d, rest)) => Value::Array((0..d).map(|_| nest(rest, data)).collect()),
    }
}

// ---------------------------------------------------------------------------
// tensor.load — safetensors sidecar ingest
// ---------------------------------------------------------------------------
//
// `tensor.load(path[, name])` reads a safetensors file — the out-of-band
// weight channel `tensor.from_list` cannot be (see
// host:project/reports/tensor_load_spike.md): 8-byte LE header length +
// JSON header `{name: {dtype, shape, data_offsets}}` + raw little-endian
// payload.  Shape comes from the file header, never the call site.
// Reads are ranged — header plus only the requested tensor's payload
// slice, decoded in chunks — so a multi-GB sidecar never lands in RAM
// whole (llama-scale verify-runs: see host's llama_e2e_retry.md).
//
// Path resolution: absolute paths pass through; relative paths resolve
// against `$TENSOR_LOAD_DIR` when set (the stdin-eval wrinkle —
// `hnl-run -` has no artifact directory, so emit keeps `.hnl` +
// `.safetensors` relocatable as a pair by exporting the var), else
// against the process cwd.
//
// dtype: `"F64"` is the identity (the internal repr is f64); `"F32"`
// widens `f32 → f64` on ingest (whisper weights are f32 — the sidecar
// halves the binary).  Every other dtype (`F16`/`BF16`/`I64`/`BOOL`/…)
// is a named error; extending is a match arm, not a format change.
// Non-finite elements ride the binary channel natively and re-encode
// through [`encode_tensor`]'s `__nonfinite__` tags.

/// Env var prepended to relative `tensor.load` paths (see module docs).
const LOAD_BASE_ENV: &str = "TENSOR_LOAD_DIR";

/// The Rust handl for `op "tensor.load" : Value... -> Tensor` —
/// `load(path)` / `load(path, name)` over a safetensors sidecar.
#[derive(Debug, Clone, Copy)]
pub struct Load;
impl Function for Load {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        require_arg_range(args, 1, 2, "tensor.load")?;
        let path = args[0]
            .as_str()
            .ok_or_else(|| type_mismatch("path: string", "non-string"))?;
        let name = match args.get(1) {
            Some(v) if !v.is_null() => Some(
                v.as_str()
                    .ok_or_else(|| type_mismatch("name: string", "non-string"))?,
            ),
            _ => None,
        };
        Ok(load_safetensors_cached(path, name)?)
    }
}

/// Resolve `path` for reading: absolute as-is; relative against
/// `$TENSOR_LOAD_DIR` when set and non-empty, else the process cwd.
fn resolve_load_path(path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    if let Ok(base) = std::env::var(LOAD_BASE_ENV) {
        if !base.is_empty() {
            return Path::new(&base).join(p);
        }
    }
    p.to_path_buf()
}

/// One tensor entry decoded from the safetensors header.
struct StEntry {
    dtype: String,
    shape: Vec<usize>,
    /// Payload-relative `[begin, end)` byte range.
    begin: usize,
    end: usize,
}

/// `tensor.load` result cache — `HOST_TENSOR_CACHE`.
///
/// `.hnl` artifacts re-evaluate the whole program per call, and every
/// eval re-runs each `tensor.load`: the ranged reads are cheap per
/// tensor but the header parse + payload read + (for ≥4096-element
/// tensors) the `__tensor_bin__` base64 encode repeat for weights that
/// are, by safetensors convention, immutable.  With
/// `HOST_TENSOR_CACHE=1` the *encoded wire `Value`* is memoized per
/// process — a hit is one `Value` clone (the base64 string memcpy)
/// instead of open+stat+header+read+encode.  Default OFF: unset,
/// empty, or `0`/`false`/`no`/`off` (case-insensitive) all leave the
/// read-every-call behavior unchanged.
///
/// The key carries the resolved path, the requested tensor name, and
/// the file's `(len, mtime)` — cheap insurance against a sidecar
/// being rewritten in place (same path, different bytes → different
/// key → reload).  Cached values are the immutable wire records, so
/// sharing them across calls is sound; the map is bounded (cleared
/// past [`LOAD_CACHE_CAP`]) and mutex-guarded for multi-threaded eval.
const LOAD_CACHE_ENV: &str = "HOST_TENSOR_CACHE";

/// Entry cap — weight sidecars bind tens-to-hundreds of tensors; past
/// this the map is cleared wholesale rather than LRU-evicting (a
/// partial clear would be just as correct, simpler is fine).
const LOAD_CACHE_CAP: usize = 1024;

/// `(resolved path, tensor name, file len, file mtime)` — see the
/// [`load_safetensors_cached`] docs for why each component is there.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LoadCacheKey {
    path: PathBuf,
    name: Option<String>,
    len: u64,
    mtime: Option<SystemTime>,
}

fn load_cache() -> &'static Mutex<HashMap<LoadCacheKey, Value>> {
    static CACHE: OnceLock<Mutex<HashMap<LoadCacheKey, Value>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Whether `HOST_TENSOR_CACHE` opts in — see the module note above.
fn load_cache_enabled() -> bool {
    match std::env::var(LOAD_CACHE_ENV) {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        ),
        Err(_) => false,
    }
}

/// `tensor.load` entry point: the cached wrapper around
/// [`load_safetensors`].  With the cache off (the default) this is a
/// straight pass-through.  With it on, the file is stat'ed first —
/// the `(len, mtime)` pair keys the entry — and a hit returns the
/// memoized wire `Value` without touching the payload again.  Stat
/// failures fall through to the uncached path so error messages stay
/// canonical.
fn load_safetensors_cached(path: &str, name: Option<&str>) -> Result<Value, EvalError> {
    if !load_cache_enabled() {
        return load_safetensors(path, name);
    }
    let resolved = resolve_load_path(path);
    let Ok(meta) = std::fs::metadata(&resolved) else {
        return load_safetensors(path, name);
    };
    let key = LoadCacheKey {
        path: resolved.clone(),
        name: name.map(str::to_string),
        len: meta.len(),
        mtime: meta.modified().ok(),
    };
    if let Some(v) = load_cache().lock().unwrap().get(&key) {
        return Ok(v.clone());
    }
    let v = load_safetensors_at(&resolved, path, name)?;
    let mut cache = load_cache().lock().unwrap();
    if cache.len() >= LOAD_CACHE_CAP {
        cache.clear();
    }
    cache.insert(key, v.clone());
    Ok(v)
}

/// Read the safetensors file at `path`, returning the tensor named
/// `name` (or the file's only tensor when `name` is `None`) as the
/// `{"__tensor__": …}` wire record.
///
/// Memory profile: only the 8-byte length prefix, the JSON header, and
/// the *requested tensor's* payload range are read — never the whole
/// file.  A llama-scale sidecar is one multi-GB file holding hundreds
/// of tensors; `std::fs::read` of it per call was the dominant
/// transient (the allocator retains the arena, so RSS stayed at
/// file-size after the first `tensor.load`).
///
/// ADR-0179: tensors with `numel >= TENSOR_BIN_THRESHOLD` skip the
/// decode entirely — the safetensors payload range *is* little-endian
/// `F32`/`F64`, so the `__tensor_bin__` record is emitted straight from
/// the ranged read (base64 of the raw bytes, `dtype: "f32"|"f64"`).
/// Smaller tensors keep the decode→`encode_tensor` path.
fn load_safetensors(path: &str, name: Option<&str>) -> Result<Value, EvalError> {
    let resolved = resolve_load_path(path);
    load_safetensors_at(&resolved, path, name)
}

/// `resolved` is the [`resolve_load_path`] output — factored out so the
/// [`load_safetensors_cached`] fast path can stat-then-key without
/// resolving twice.
fn load_safetensors_at(resolved: &Path, path: &str, name: Option<&str>) -> Result<Value, EvalError> {
    let err = |msg: String| EvalError::Other(format!("tensor.load: {path}: {msg}"));
    let mut file = std::fs::File::open(resolved).map_err(|e| {
        EvalError::Other(format!(
            "tensor.load: cannot read {path} (resolved {}): {e}",
            resolved.display()
        ))
    })?;
    let file_len = file
        .metadata()
        .map_err(|e| err(format!("cannot stat {path}: {e}")))?
        .len();

    let mut lenbuf = [0u8; 8];
    file.read_exact(&mut lenbuf)
        .map_err(|_| err("truncated — no 8-byte header length".to_string()))?;
    let hlen = u64::from_le_bytes(lenbuf);
    let hlen =
        usize::try_from(hlen).map_err(|_| err("header length overflows usize".to_string()))?;
    // The header must fit inside the file — check BEFORE allocating so a
    // corrupt length prefix cannot ask for a multi-GB buffer.
    let hend = 8u64
        .checked_add(hlen as u64)
        .filter(|&e| e <= file_len)
        .ok_or_else(|| err("truncated header".to_string()))?;
    let mut hbuf = vec![0u8; hlen];
    file.read_exact(&mut hbuf)
        .map_err(|_| err("truncated header".to_string()))?;

    let entries = parse_header(&hbuf, path)?;
    let chosen = choose_entry(&entries, name, path)?;

    let payload_len = file_len - hend;
    if chosen.end as u64 > payload_len || chosen.begin > chosen.end {
        return Err(err(format!(
            "data_offsets {:?} out of bounds (payload {} bytes)",
            [chosen.begin, chosen.end],
            payload_len
        )));
    }
    // Validate dtype + payload size up front (the same check
    // `entry_decode_plan` applies on the decode path) — unsupported
    // dtypes stay named errors before any payload read.
    let (elem_size, _, expected) = entry_decode_plan(chosen, path)?;
    let nbytes = chosen.end - chosen.begin;
    if nbytes != expected {
        return Err(err(format!(
            "payload {} bytes != {} elements × {elem_size}B for shape {:?}",
            nbytes,
            chosen.shape.iter().product::<usize>(),
            chosen.shape
        )));
    }
    file.seek(SeekFrom::Start(hend + chosen.begin as u64))
        .map_err(|e| err(format!("cannot seek to tensor payload: {e}")))?;

    let numel: usize = chosen.shape.iter().product();
    if numel >= handl_value::TENSOR_BIN_THRESHOLD {
        // ADR-0179 large path: the payload bytes ARE the wire payload —
        // read the range once and emit the `__tensor_bin__` record
        // without decoding (no f32→f64 widen, no JSON array).
        let mut payload = vec![0u8; nbytes];
        file.read_exact(&mut payload)
            .map_err(|e| err(format!("cannot read tensor payload: {e}")))?;
        let wire_dtype = match chosen.dtype.as_str() {
            "F64" => "f64",
            "F32" => "f32",
            // entry_decode_plan already rejected every other dtype.
            other => return Err(err(format!("unsupported dtype {other:?}"))),
        };
        return handl_value::tensor_bin_record(&chosen.shape, wire_dtype, &payload)
            .ok_or_else(|| err("internal: __tensor_bin__ record encode failed".to_string()));
    }
    decode_entry_streaming(&mut file, chosen, path).map(encode_tensor)
}

/// Parse safetensors bytes: `u64le header_len`, JSON header, raw LE
/// payload.  All failures are `EvalError::Other` naming `path` + reason.
/// Test-only: the op path goes through [`load_safetensors`]'s ranged
/// reads, which never materialize the whole file.
#[cfg(test)]
fn parse_safetensors(bytes: &[u8], path: &str, name: Option<&str>) -> Result<Tensor, EvalError> {
    let err = |msg: String| EvalError::Other(format!("tensor.load: {path}: {msg}"));
    if bytes.len() < 8 {
        return Err(err("truncated — no 8-byte header length".to_string()));
    }
    let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let hend = 8usize
        .checked_add(hlen)
        .filter(|&e| e <= bytes.len())
        .ok_or_else(|| err("truncated header".to_string()))?;
    let entries = parse_header(&bytes[8..hend], path)?;
    let chosen = choose_entry(&entries, name, path)?;

    let payload = &bytes[hend..];
    if chosen.end > payload.len() || chosen.begin > chosen.end {
        return Err(err(format!(
            "data_offsets {:?} out of bounds (payload {} bytes)",
            [chosen.begin, chosen.end],
            payload.len()
        )));
    }
    decode_entry(chosen, &payload[chosen.begin..chosen.end], path)
}

/// Parse the JSON header object into `(name, entry)` pairs — every
/// non-`__metadata__` key is a tensor entry.
fn parse_header(header_bytes: &[u8], path: &str) -> Result<Vec<(String, StEntry)>, EvalError> {
    let err = |msg: String| EvalError::Other(format!("tensor.load: {path}: {msg}"));
    let header: Value = serde_json::from_slice(header_bytes)
        .map_err(|e| err(format!("header is not valid JSON: {e}")))?;
    let obj = header
        .as_object()
        .ok_or_else(|| err("header is not a JSON object".to_string()))?;

    let mut entries: Vec<(String, StEntry)> = Vec::with_capacity(obj.len());
    for (key, val) in obj {
        if key == "__metadata__" {
            continue;
        }
        let rec = val
            .as_object()
            .ok_or_else(|| err(format!("header entry {key:?} is not an object")))?;
        let dtype = rec
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| err(format!("header entry {key:?}: missing dtype")))?
            .to_string();
        let shape_val = rec
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| err(format!("header entry {key:?}: missing shape array")))?;
        let mut shape = Vec::with_capacity(shape_val.len());
        for d in shape_val {
            let n = d
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| err(format!("header entry {key:?}: bad shape extent")))?;
            shape.push(n);
        }
        let offs = rec
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| err(format!("header entry {key:?}: missing data_offsets")))?;
        if offs.len() != 2 {
            return Err(err(format!(
                "header entry {key:?}: data_offsets must be [begin, end]"
            )));
        }
        let begin = offs[0]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| err(format!("header entry {key:?}: bad data_offsets")))?;
        let end = offs[1]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| err(format!("header entry {key:?}: bad data_offsets")))?;
        entries.push((
            key.clone(),
            StEntry {
                dtype,
                shape,
                begin,
                end,
            },
        ));
    }
    Ok(entries)
}

/// Select the requested entry: `name`, or the file's only tensor when
/// `name` is `None`.
fn choose_entry<'a>(
    entries: &'a [(String, StEntry)],
    name: Option<&str>,
    path: &str,
) -> Result<&'a StEntry, EvalError> {
    let err = |msg: String| EvalError::Other(format!("tensor.load: {path}: {msg}"));
    match name {
        Some(want) => entries
            .iter()
            .find(|(k, _)| k == want)
            .map(|(_, e)| e)
            .ok_or_else(|| {
                let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
                err(format!("no tensor named {want:?} — file holds {keys:?}"))
            }),
        None => match entries {
            [(_, only)] => Ok(only),
            [] => Err(err("file holds no tensors".to_string())),
            _ => {
                let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
                Err(err(format!(
                    "file holds {} tensors {keys:?} — pass a name",
                    keys.len()
                )))
            }
        },
    }
}

/// Decode one tensor's payload bytes (`chosen.begin..chosen.end`,
/// already bounds-checked by the caller) into the f64 `Tensor`.
#[cfg(test)]
fn decode_entry(chosen: &StEntry, bytes_of: &[u8], path: &str) -> Result<Tensor, EvalError> {
    let err = |msg: String| EvalError::Other(format!("tensor.load: {path}: {msg}"));
    let (elem_size, decode, expected) = entry_decode_plan(chosen, path)?;
    if bytes_of.len() != expected {
        return Err(err(format!(
            "payload {} bytes != {} elements × {elem_size}B for shape {:?}",
            bytes_of.len(),
            chosen.shape.iter().product::<usize>(),
            chosen.shape
        )));
    }
    let data: Vec<f64> = bytes_of.chunks_exact(elem_size).map(decode).collect();
    Ok(Tensor {
        shape: chosen.shape.clone(),
        data,
    })
}

/// Stream-decode the payload range the file cursor sits at: read +
/// decode in fixed-size chunks so the transient stays O(chunk +
/// output), never O(tensor payload).  The range length is validated
/// against `shape × dtype` before the first read, same check
/// [`decode_entry`] applies to a materialized slice.
fn decode_entry_streaming(
    file: &mut std::fs::File,
    chosen: &StEntry,
    path: &str,
) -> Result<Tensor, EvalError> {
    /// Read granularity — a multiple of both supported element sizes.
    const CHUNK: usize = 1 << 20;
    let err = |msg: String| EvalError::Other(format!("tensor.load: {path}: {msg}"));
    let (elem_size, decode, expected) = entry_decode_plan(chosen, path)?;
    let nbytes = chosen.end - chosen.begin;
    if nbytes != expected {
        return Err(err(format!(
            "payload {} bytes != {} elements × {elem_size}B for shape {:?}",
            nbytes,
            chosen.shape.iter().product::<usize>(),
            chosen.shape
        )));
    }
    let numel: usize = chosen.shape.iter().product();
    let mut data: Vec<f64> = Vec::with_capacity(numel);
    let mut buf = vec![0u8; CHUNK];
    let mut remaining = expected;
    while remaining > 0 {
        // `remaining` and `CHUNK` are both multiples of `elem_size`,
        // so `take` always is too.
        let take = remaining.min(CHUNK);
        file.read_exact(&mut buf[..take])
            .map_err(|e| err(format!("cannot read tensor payload: {e}")))?;
        for c in buf[..take].chunks_exact(elem_size) {
            data.push(decode(c));
        }
        remaining -= take;
    }
    Ok(Tensor {
        shape: chosen.shape.clone(),
        data,
    })
}

/// Resolve the entry's dtype/shape to `(elem_size, decoder, expected
/// payload bytes)` — the shared validation both decode paths apply.
fn entry_decode_plan(
    chosen: &StEntry,
    path: &str,
) -> Result<(usize, fn(&[u8]) -> f64, usize), EvalError> {
    let err = |msg: String| EvalError::Other(format!("tensor.load: {path}: {msg}"));
    let numel: usize = chosen.shape.iter().product();
    let (elem_size, decode): (usize, fn(&[u8]) -> f64) = match chosen.dtype.as_str() {
        "F64" => (8, |b| f64::from_le_bytes(b.try_into().unwrap())),
        "F32" => (4, |b| f32::from_le_bytes(b.try_into().unwrap()) as f64),
        other => {
            return Err(err(format!(
                "unsupported dtype {other:?} — tensor.load accepts \"F64\" and \"F32\""
            )));
        }
    };
    let expected = numel
        .checked_mul(elem_size)
        .ok_or_else(|| err("shape extent overflows usize".to_string()))?;
    Ok((elem_size, decode, expected))
}

/// The Rust handl for `op "tensor.shape" : Tensor -> (List Int)`.
#[derive(Debug, Clone, Copy)]
pub struct Shape;
impl Function for Shape {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 1)?;
        let t = require_tensor(args, 0)?;
        Ok(json!(t.shape))
    }
}
// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::testutil::{call_fn, tensor};
    use serde_json::json;

    #[test]
    fn decode_round_trip() {
        let v = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = decode_tensor(&v).unwrap();
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(encode_tensor(t.clone()), v);
    }

    #[test]
    fn decode_dtype_optional_on_input() {
        let v = json!({"__tensor__": {"shape": [1], "data": [7]}});
        let t = decode_tensor(&v).unwrap();
        assert_eq!(t.data, vec![7.0]);
    }

    #[test]
    fn decode_rejects_wrong_dtype() {
        let v = json!({"__tensor__": {"shape": [1], "data": [7], "dtype": "i32"}});
        assert!(decode_tensor(&v).is_err());
    }

    #[test]
    fn decode_rejects_untagged_and_malformed() {
        assert!(decode_tensor(&json!({"shape": [1], "data": [1]})).is_err());
        assert!(decode_tensor(&json!({"__tensor__": {"shape": [2], "data": [1]}})).is_err());
        assert!(decode_tensor(&json!({"__tensor__": {"shape": [-1], "data": []}})).is_err());
        assert!(decode_tensor(&json!(42)).is_err());
    }

    // ── non-finite wire elements ──────────────────────────────────────

    /// The `{"__nonfinite__": …}` spelling of each non-finite class.
    fn nonfinite(tag: &str) -> Value {
        json!({"__nonfinite__": tag})
    }

    #[test]
    fn nonfinite_elements_round_trip() {
        let v = json!({"__tensor__": {
            "shape": [2, 3],
            "data": [1.0, nonfinite("nan"), 2.5, nonfinite("inf"), nonfinite("-inf"), -0.0],
            "dtype": "f64",
        }});
        let t = decode_tensor(&v).unwrap();
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.data[0], 1.0);
        assert!(t.data[1].is_nan());
        assert_eq!(t.data[2], 2.5);
        assert_eq!(t.data[3], f64::INFINITY);
        assert_eq!(t.data[4], f64::NEG_INFINITY);
        assert_eq!(t.data[5], -0.0);
        // Encode reproduces the tagged spelling exactly.
        assert_eq!(encode_tensor(t), v);
    }

    #[test]
    fn encode_tags_every_nonfinite_class() {
        let t = Tensor {
            shape: vec![4],
            data: vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.5],
        };
        let v = encode_tensor(t);
        assert_eq!(v["__tensor__"]["data"][0], nonfinite("nan"));
        assert_eq!(v["__tensor__"]["data"][1], nonfinite("inf"));
        assert_eq!(v["__tensor__"]["data"][2], nonfinite("-inf"));
        assert_eq!(v["__tensor__"]["data"][3], json!(1.5));
    }

    #[test]
    fn decode_rejects_bad_nonfinite_tags_and_nulls() {
        for bad in [
            json!(null),
            json!("nan"),
            json!({"__nonfinite__": "bogus"}),
            json!({"__nonfinite__": 1}),
            json!({"other": "nan"}),
        ] {
            let v = json!({"__tensor__": {"shape": [1], "data": [bad]}});
            assert!(decode_tensor(&v).is_err(), "must reject {v}");
        }
    }

    #[test]
    fn from_list_accepts_nonfinite_tags() {
        let v = call_fn(
            &FromList,
            &[json!([[1, {"__nonfinite__": "inf"}], [{"__nonfinite__": "nan"}, -2]])],
        )
        .unwrap();
        assert_eq!(v["__tensor__"]["shape"], json!([2, 2]));
        let t = decode_tensor(&v).unwrap();
        assert_eq!(t.data[0], 1.0);
        assert_eq!(t.data[1], f64::INFINITY);
        assert!(t.data[2].is_nan());
        assert_eq!(t.data[3], -2.0);
    }

    #[test]
    fn to_list_emits_nonfinite_tags() {
        // (`tensor()` builds wire via `json!`, which would null the
        // non-finites — spell the tagged form directly.)
        let wire = json!({"__tensor__": {
            "shape": [2],
            "data": [nonfinite("inf"), nonfinite("nan")],
            "dtype": "f64",
        }});
        let v = call_fn(&ToList, &[wire]).unwrap();
        assert_eq!(v, json!([nonfinite("inf"), nonfinite("nan")]));
        // And the pair round-trips: from_list(to_list(x)) == x.
        let back = call_fn(&FromList, &[v]).unwrap();
        let t = decode_tensor(&back).unwrap();
        assert_eq!(t.data[0], f64::INFINITY);
        assert!(t.data[1].is_nan());
    }

    // ── construction ──────────────────────────────────────────────────

    #[test]
    fn zeros_fills() {
        let v = call_fn(&Zeros, &[json!([2, 2])]).unwrap();
        assert_eq!(v, tensor(&[2, 2], &[0.0; 4]));
    }

    #[test]
    fn ones_fills() {
        let v = call_fn(&Ones, &[json!([3])]).unwrap();
        assert_eq!(v, tensor(&[3], &[1.0; 3]));
    }

    #[test]
    fn zeros_scalar_shape_is_rank_zero() {
        let v = call_fn(&Zeros, &[json!([])]).unwrap();
        assert_eq!(v, tensor(&[], &[0.0]));
    }

    #[test]
    fn zeros_rejects_negative_extent() {
        assert!(call_fn(&Zeros, &[json!([2, -1])]).is_err());
    }

    #[test]
    fn from_list_infers_shape() {
        let v = call_fn(&FromList, &[json!([[1, 2, 3], [4, 5, 6]])]).unwrap();
        assert_eq!(v, tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    }

    #[test]
    fn from_list_rank_one() {
        let v = call_fn(&FromList, &[json!([1, 2])]).unwrap();
        assert_eq!(v, tensor(&[2], &[1.0, 2.0]));
    }

    #[test]
    fn from_list_rejects_ragged() {
        assert!(call_fn(&FromList, &[json!([[1, 2], [3]])]).is_err());
        assert!(call_fn(&FromList, &[json!([[1], 2])]).is_err());
        assert!(call_fn(&FromList, &[json!([1, "x"])]).is_err());
        assert!(call_fn(&FromList, &[json!(5)]).is_err());
    }

    // ── projection ────────────────────────────────────────────────────

    #[test]
    fn to_list_nests() {
        let v = call_fn(&ToList, &[tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]).unwrap();
        assert_eq!(v, json!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
    }

    #[test]
    fn from_list_to_list_round_trip() {
        let input = json!([[1, 2], [3, 4], [5, 6]]);
        let t = call_fn(&FromList, &[input.clone()]).unwrap();
        let back = call_fn(&ToList, &[t]).unwrap();
        // `to_list` emits `f64`s — semantically equal to the integer input
        // (handl `=` is Int/Float cross-comparable) though the JSON
        // `Number` representation differs.
        assert_eq!(back, json!([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]));
    }

    #[test]
    fn shape_projects() {
        let v = call_fn(&Shape, &[tensor(&[2, 3], &[0.0; 6])]).unwrap();
        assert_eq!(v, json!([2, 3]));
    }

    // ── tensor.load — safetensors sidecar ─────────────────────────────

    /// Serialize `(name, dtype, shape, payload-bytes)` entries into
    /// safetensors wire form (8-byte LE header len + JSON + payload).
    fn safetensors(tensors: &[(&str, &str, &[usize], Vec<u8>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let begin = payload.len();
            payload.extend_from_slice(bytes);
            let end = payload.len();
            header.insert(
                (*name).to_string(),
                json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [begin, end],
                }),
            );
        }
        let hjson = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut out = (hjson.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&hjson);
        out.extend_from_slice(&payload);
        out
    }

    fn f64s(xs: &[f64]) -> Vec<u8> {
        xs.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn f32s(xs: &[f32]) -> Vec<u8> {
        xs.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// Write `bytes` under a per-test temp dir; returns the file path.
    fn write_st(test: &str, bytes: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "host-tensor-load-test-{}-{test}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.safetensors");
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn load_f64_single_tensor_unnamed() {
        let p = write_st(
            "f64",
            &safetensors(&[("w", "F64", &[2, 2], f64s(&[1.0, 2.0, 3.0, 4.0]))]),
        );
        let v = call_fn(&Load, &[json!(p.to_string_lossy())]).unwrap();
        assert_eq!(v, tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]));
        // An explicit `null` name selects the same way as omission.
        let v = call_fn(&Load, &[json!(p.to_string_lossy()), Value::Null]).unwrap();
        assert_eq!(v, tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]));
        std::fs::remove_dir_all(p.parent().unwrap()).ok();
    }

    #[test]
    fn load_named_from_multi_tensor_file() {
        let p = write_st(
            "multi",
            &safetensors(&[
                ("a", "F64", &[2], f64s(&[1.0, 2.0])),
                ("b", "F64", &[3], f64s(&[3.0, 4.0, 5.0])),
            ]),
        );
        let path = p.to_string_lossy().to_string();
        let v = call_fn(&Load, &[json!(path), json!("b")]).unwrap();
        assert_eq!(v, tensor(&[3], &[3.0, 4.0, 5.0]));
        std::fs::remove_dir_all(p.parent().unwrap()).ok();
    }

    #[test]
    fn load_unnamed_multi_tensor_errors_naming_keys() {
        let p = write_st(
            "multi-err",
            &safetensors(&[
                ("a", "F64", &[1], f64s(&[1.0])),
                ("b", "F64", &[1], f64s(&[2.0])),
            ]),
        );
        let err = call_fn(&Load, &[json!(p.to_string_lossy())]).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("tensor.load"), "{msg}");
        assert!(msg.contains("\"a\"") && msg.contains("\"b\""), "{msg}");
        std::fs::remove_dir_all(p.parent().unwrap()).ok();
    }

    #[test]
    fn load_f32_widens_to_f64() {
        let p = write_st(
            "f32",
            &safetensors(&[("w", "F32", &[3], f32s(&[0.5, -1.25, 3.0]))]),
        );
        let v = call_fn(&Load, &[json!(p.to_string_lossy())]).unwrap();
        assert_eq!(v, tensor(&[3], &[0.5, -1.25, 3.0]));
        std::fs::remove_dir_all(p.parent().unwrap()).ok();
    }

    #[test]
    fn load_rejects_unsupported_dtype() {
        for dtype in ["I64", "F16", "BF16", "BOOL"] {
            let p = write_st(
                &format!("dtype-{dtype}"),
                &safetensors(&[("w", dtype, &[1], f64s(&[1.0]))]),
            );
            let err = call_fn(&Load, &[json!(p.to_string_lossy())]).unwrap_err();
            assert!(format!("{err}").contains(dtype), "{err}");
            std::fs::remove_dir_all(p.parent().unwrap()).ok();
        }
    }

    #[test]
    fn load_missing_file_names_path() {
        let err = call_fn(&Load, &[json!("/nonexistent/dir/nope.safetensors")]).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("tensor.load"), "{msg}");
        assert!(msg.contains("nope.safetensors"), "{msg}");
    }

    #[test]
    fn load_missing_name_errors() {
        let p = write_st(
            "missing-name",
            &safetensors(&[("w", "F64", &[1], f64s(&[1.0]))]),
        );
        let err = call_fn(&Load, &[json!(p.to_string_lossy()), json!("nope")]).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("nope"), "{msg}");
        assert!(msg.contains("\"w\""), "{msg}");
        std::fs::remove_dir_all(p.parent().unwrap()).ok();
    }

    #[test]
    fn load_nonfinite_elements_tag_on_wire() {
        let p = write_st(
            "nonfinite",
            &safetensors(&[("w", "F64", &[3], f64s(&[f64::NAN, f64::INFINITY, 1.0]))]),
        );
        let v = call_fn(&Load, &[json!(p.to_string_lossy())]).unwrap();
        assert_eq!(v["__tensor__"]["data"][0], nonfinite("nan"));
        assert_eq!(v["__tensor__"]["data"][1], nonfinite("inf"));
        // And the decoded tensor carries the real values.
        let t = decode_tensor(&v).unwrap();
        assert!(t.data[0].is_nan());
        assert_eq!(t.data[1], f64::INFINITY);
        std::fs::remove_dir_all(p.parent().unwrap()).ok();
    }

    #[test]
    fn load_tensor_load_dir_env_base() {
        let dir = std::env::temp_dir().join(format!("host-tensor-load-env-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("rel.safetensors"),
            safetensors(&[("w", "F64", &[2], f64s(&[7.0, 8.0]))]),
        )
        .unwrap();
        // SAFETY: single-threaded w.r.t. other `tensor.load` tests —
        // every other test passes an absolute path, which bypasses
        // `TENSOR_LOAD_DIR` entirely.
        unsafe { std::env::set_var("TENSOR_LOAD_DIR", &dir) };
        let v = call_fn(&Load, &[json!("rel.safetensors")]).unwrap();
        unsafe { std::env::remove_var("TENSOR_LOAD_DIR") };
        assert_eq!(v, tensor(&[2], &[7.0, 8.0]));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_bad_args_and_truncated_files() {
        // Arity / arg types.
        assert!(call_fn(&Load, &[]).is_err());
        assert!(call_fn(&Load, &[json!(42)]).is_err());
        assert!(call_fn(&Load, &[json!("x"), json!(7)]).is_err());
        // Truncated / non-safetensors bytes.
        let p = write_st("trunc", &[0u8; 4]);
        assert!(call_fn(&Load, &[json!(p.to_string_lossy())]).is_err());
        let p2 = write_st("badjson", &[0u8; 8]);
        assert!(call_fn(&Load, &[json!(p2.to_string_lossy())]).is_err());
        std::fs::remove_dir_all(p.parent().unwrap()).ok();
        std::fs::remove_dir_all(p2.parent().unwrap()).ok();
        // Direct parse checks: header len beyond EOF, payload OOB.
        assert!(parse_safetensors(&[0xff; 8], "mem", None).is_err());
        let mut bad = (4u64).to_le_bytes().to_vec();
        bad.extend_from_slice(b"null"); // valid JSON, not an object
        assert!(parse_safetensors(&bad, "mem", None).is_err());
    }

    // ── ADR-0179 binary `__tensor_bin__` wire spelling ────────────────

    /// LE bytes for a slice of f32/f64 (the safetensors payload
    /// spelling `__tensor_bin__` carries).
    fn le_bytes_f32(xs: &[f32]) -> Vec<u8> {
        xs.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    fn le_bytes_f64(xs: &[f64]) -> Vec<u8> {
        xs.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn decode_bin_f32_widens_to_f64() {
        let rec =
            handl_value::tensor_bin_record(&[2, 2], "f32", &le_bytes_f32(&[1.5, -2.0, 0.5, 4.0]))
                .unwrap();
        let t = decode_tensor(&rec).unwrap();
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.data, vec![1.5, -2.0, 0.5, 4.0]);
    }

    #[test]
    fn decode_bin_f64_preserves_values() {
        let rec =
            handl_value::tensor_bin_record(&[3], "f64", &le_bytes_f64(&[f64::NAN, 2.0, -0.0]))
                .unwrap();
        let t = decode_tensor(&rec).unwrap();
        assert!(t.data[0].is_nan());
        assert_eq!(t.data[1], 2.0);
        assert_eq!(t.data[2].to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn decode_rejects_malformed_bin() {
        // Bad base64.
        let bad_b64 = json!({"__tensor__": {"shape": [2], "dtype": "f32",
            "data": {"__tensor_bin__": "!!!"}}});
        assert!(matches!(
            decode_tensor(&bad_b64),
            Err(EvalError::TypeMismatch(_))
        ));
        // Payload length != numel × itemsize (a [2] record relabeled [4]).
        let short = handl_value::tensor_bin_record(&[2], "f32", &le_bytes_f32(&[1.0, 2.0]))
            .map(|mut v| {
                v["__tensor__"]["shape"] = json!([4]);
                v
            })
            .unwrap();
        assert!(decode_tensor(&short).is_err());
        // dtype absent — the binary spelling requires it.
        let no_dtype = json!({"__tensor__": {"shape": [2],
            "data": {"__tensor_bin__": "AAAAAAAAAAA="}}});
        assert!(decode_tensor(&no_dtype).is_err());
        // Unsupported dtype on the bin spelling.
        let bad_dtype = json!({"__tensor__": {"shape": [2], "dtype": "i32",
            "data": {"__tensor_bin__": "AAAAAAAAAAA="}}});
        assert!(decode_tensor(&bad_dtype).is_err());
        // "f32" + array data is malformed (the array spelling is f64).
        let f32_array = json!({"__tensor__": {"shape": [2], "dtype": "f32", "data": [1.0, 2.0]}});
        assert!(decode_tensor(&f32_array).is_err());
    }

    #[test]
    fn encode_emits_bin_at_threshold() {
        let n = handl_value::TENSOR_BIN_THRESHOLD;
        let big = Tensor {
            shape: vec![n],
            data: vec![1.5; n],
        };
        let v = encode_tensor(big.clone());
        assert_eq!(v["__tensor__"]["dtype"], json!("f64"));
        assert!(v["__tensor__"]["data"]["__tensor_bin__"].is_string());
        // And it decodes back to the same tensor.
        assert_eq!(decode_tensor(&v).unwrap(), big);
        // Below the threshold the array form is unchanged.
        let small = Tensor {
            shape: vec![n - 1],
            data: vec![1.5; n - 1],
        };
        let v = encode_tensor(small);
        assert!(v["__tensor__"]["data"].is_array());
    }

    #[test]
    fn load_large_tensor_returns_bin_record() {
        // numel >= TENSOR_BIN_THRESHOLD → the wire record carries the
        // raw safetensors payload as __tensor_bin__ (no decode): an
        // F32 sidecar keeps dtype "f32".
        let n = handl_value::TENSOR_BIN_THRESHOLD;
        let xs: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let p = write_st(
            "big-f32",
            &safetensors(&[("w", "F32", &[n], le_bytes_f32(&xs))]),
        );
        let v = call_fn(&Load, &[json!(p.to_string_lossy())]).unwrap();
        assert_eq!(v["__tensor__"]["dtype"], json!("f32"));
        assert!(v["__tensor__"]["data"]["__tensor_bin__"].is_string());
        // The record decodes through the same codec every op uses.
        let t = decode_tensor(&v).unwrap();
        assert_eq!(t.shape, vec![n]);
        assert_eq!(t.data.len(), n);
        assert_eq!(t.data[0], 0.0);
        assert_eq!(t.data[1], 0.5);
        assert_eq!(t.data[n - 1], (n - 1) as f32 as f64 * 0.5);
        std::fs::remove_dir_all(p.parent().unwrap()).ok();
    }

    #[test]
    fn load_small_tensor_stays_array_form() {
        // Below the threshold tensor.load is byte-identical to before:
        // decode → f64 array wire record.
        let p = write_st(
            "small-f32",
            &safetensors(&[("w", "F32", &[3], f32s(&[0.5, -1.25, 3.0]))]),
        );
        let v = call_fn(&Load, &[json!(p.to_string_lossy())]).unwrap();
        assert_eq!(v["__tensor__"]["dtype"], json!("f64"));
        assert!(v["__tensor__"]["data"].is_array());
        assert_eq!(v, tensor(&[3], &[0.5, -1.25, 3.0]));
        std::fs::remove_dir_all(p.parent().unwrap()).ok();
    }

    // ── HOST_TENSOR_CACHE — the tensor.load result cache ─────────────

    /// `HOST_TENSOR_CACHE` is process-global; serialize the tests that
    /// mutate it (the `TENSOR_LOAD_DIR` test above gets away without a
    /// lock only because no other test reads that var).
    static CACHE_ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Set `HOST_TENSOR_CACHE` for the test body; the guard restores
    /// the prior state on drop.
    struct CacheEnv(Option<String>);
    impl CacheEnv {
        fn set(v: &str) -> Self {
            let prev = std::env::var(LOAD_CACHE_ENV).ok();
            // SAFETY: serialized by CACHE_ENV_LOCK — no other test
            // thread reads or writes HOST_TENSOR_CACHE concurrently.
            unsafe { std::env::set_var(LOAD_CACHE_ENV, v) };
            Self(prev)
        }
    }
    impl Drop for CacheEnv {
        fn drop(&mut self) {
            // SAFETY: same serialization argument as `set`.
            unsafe {
                match &self.0 {
                    Some(prev) => std::env::set_var(LOAD_CACHE_ENV, prev),
                    None => std::env::remove_var(LOAD_CACHE_ENV),
                }
            }
        }
    }

    #[test]
    fn load_cache_off_by_default() {
        let _g = CACHE_ENV_LOCK.lock().unwrap();
        let _env = CacheEnv::set("0");
        let p = write_st(
            "cache-off",
            &safetensors(&[("w", "F64", &[2], f64s(&[1.0, 2.0]))]),
        );
        let before = load_cache().lock().unwrap().len();
        let v1 = call_fn(&Load, &[json!(p.to_string_lossy())]).unwrap();
        let v2 = call_fn(&Load, &[json!(p.to_string_lossy())]).unwrap();
        assert_eq!(v1, v2);
        // Disabled: nothing was memoized.
        assert_eq!(load_cache().lock().unwrap().len(), before);
        std::fs::remove_dir_all(p.parent().unwrap()).ok();
    }

    #[test]
    fn load_cache_hits_and_invalidates_on_rewrite() {
        let _g = CACHE_ENV_LOCK.lock().unwrap();
        let _env = CacheEnv::set("1");
        let p = write_st(
            "cache-hit",
            &safetensors(&[("w", "F64", &[2], f64s(&[1.0, 2.0]))]),
        );
        let v1 = call_fn(&Load, &[json!(p.to_string_lossy())]).unwrap();
        let cached_len = load_cache().lock().unwrap().len();
        assert!(cached_len >= 1, "first load should populate the cache");
        // Second call: same key → memoized wire Value, byte-identical.
        let v2 = call_fn(&Load, &[json!(p.to_string_lossy())]).unwrap();
        assert_eq!(v1, v2);
        // Rewrite the sidecar with different content AND a different
        // length — the (len, mtime) key must miss and reload.
        std::fs::write(
            &p,
            safetensors(&[("w", "F64", &[3], f64s(&[9.0, 8.0, 7.0]))]),
        )
        .unwrap();
        let v3 = call_fn(&Load, &[json!(p.to_string_lossy())]).unwrap();
        assert_eq!(v3, tensor(&[3], &[9.0, 8.0, 7.0]));
        std::fs::remove_dir_all(p.parent().unwrap()).ok();
    }

    #[test]
    fn load_cache_named_entries_are_distinct() {
        let _g = CACHE_ENV_LOCK.lock().unwrap();
        let _env = CacheEnv::set("yes");
        let p = write_st(
            "cache-names",
            &safetensors(&[
                ("a", "F64", &[1], f64s(&[1.0])),
                ("b", "F64", &[1], f64s(&[2.0])),
            ]),
        );
        let va = call_fn(&Load, &[json!(p.to_string_lossy()), json!("a")]).unwrap();
        let vb = call_fn(&Load, &[json!(p.to_string_lossy()), json!("b")]).unwrap();
        assert_eq!(va, tensor(&[1], &[1.0]));
        assert_eq!(vb, tensor(&[1], &[2.0]));
        // Both (path, name) pairs memoized independently.
        assert!(load_cache().lock().unwrap().len() >= 2);
        std::fs::remove_dir_all(p.parent().unwrap()).ok();
    }
}
