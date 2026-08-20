//! Zero-copy access to Kernelyra's stable C ABI. Link `kernelyra_core` or load
//! the bundled binary through the application; orchestration remains in Client.

use std::ffi::{c_char, c_void, CStr};

pub const ZIG_MEMORY: u32 = 1;
pub const FORTRAN_NUMERIC: u32 = 2;
pub const ALL_COMPONENTS: u32 = 3;

#[link(name = "kernelyra_core")]
extern "C" {
    fn kr_core_version() -> *const c_char;
    fn kr_core_components() -> *const c_char;
    fn kr_core_component_mask() -> u32;
    fn kr_core_enabled_component_mask() -> u32;
    fn kr_core_set_component_mask(mask: u32) -> u32;
    fn kr_memory_alloc_aligned(bytes: usize, alignment: usize) -> *mut c_void;
    fn kr_memory_free_aligned(pointer: *mut c_void);
    fn kr_memory_normalize_f32(
        data: *mut f32,
        rows: usize,
        features: usize,
        means: *const f32,
        stds: *const f32,
    );
    fn kr_memory_copy_f32(destination: *mut f32, source: *const f32, values: usize);
    fn kr_memory_zero_f32(destination: *mut f32, values: usize);
    fn kr_numeric_gradient_f32(
        x: *const f32,
        errors: *const f32,
        rows: usize,
        features: usize,
        gradient: *mut f32,
    );
    fn kr_kernel_dot_f32(left: *const f32, right: *const f32, values: usize) -> f32;
    fn kr_model_destroy(handle: *mut c_void);
}

fn abi_text(pointer: *const c_char) -> &'static str {
    if pointer.is_null() {
        return "";
    }
    // SAFETY: Kernelyra returns process-lifetime, immutable, NUL-terminated strings.
    unsafe { CStr::from_ptr(pointer) }.to_str().unwrap_or("")
}

pub fn version() -> &'static str {
    // SAFETY: no arguments and the ABI returns a process-lifetime string.
    abi_text(unsafe { kr_core_version() })
}

pub fn components() -> &'static str {
    // SAFETY: no arguments and the ABI returns a process-lifetime string.
    abi_text(unsafe { kr_core_components() })
}

pub fn component_mask() -> u32 {
    // SAFETY: pure ABI capability query.
    unsafe { kr_core_component_mask() }
}

pub fn enabled_component_mask() -> u32 {
    // SAFETY: pure ABI dispatch query.
    unsafe { kr_core_enabled_component_mask() }
}

pub fn set_component_mask(mask: u32) -> u32 {
    // SAFETY: the core masks unknown bits and atomically updates dispatch.
    unsafe { kr_core_set_component_mask(mask) }
}

pub fn normalize(
    data: &mut [f32],
    rows: usize,
    means: &[f32],
    stds: &[f32],
) -> Result<(), &'static str> {
    let features = means.len();
    if features == 0 || stds.len() != features || data.len() != rows.saturating_mul(features) {
        return Err("invalid normalization shapes");
    }
    // SAFETY: validated slice lengths keep all pointers in bounds for the ABI call.
    unsafe {
        kr_memory_normalize_f32(
            data.as_mut_ptr(),
            rows,
            features,
            means.as_ptr(),
            stds.as_ptr(),
        )
    };
    Ok(())
}

pub fn gradient(x: &[f32], errors: &[f32], features: usize) -> Result<Vec<f32>, &'static str> {
    if features == 0 || x.len() != errors.len().saturating_mul(features) {
        return Err("invalid gradient shapes");
    }
    let mut result = vec![0.0; features];
    // SAFETY: validated shapes and owned output provide valid non-overlapping buffers.
    unsafe {
        kr_numeric_gradient_f32(
            x.as_ptr(),
            errors.as_ptr(),
            errors.len(),
            features,
            result.as_mut_ptr(),
        )
    };
    Ok(result)
}

pub fn copy(destination: &mut [f32], source: &[f32]) -> Result<(), &'static str> {
    if destination.len() != source.len() {
        return Err("copy inputs have different lengths");
    }
    // SAFETY: equally sized slices provide valid non-overlapping C ABI buffers.
    unsafe { kr_memory_copy_f32(destination.as_mut_ptr(), source.as_ptr(), source.len()) };
    Ok(())
}

pub fn zero(values: &mut [f32]) {
    // SAFETY: the mutable slice gives the ABI exclusive access to all elements.
    unsafe { kr_memory_zero_f32(values.as_mut_ptr(), values.len()) };
}

pub fn dot(left: &[f32], right: &[f32]) -> Result<f32, &'static str> {
    if left.len() != right.len() {
        return Err("dot inputs have different lengths");
    }
    // SAFETY: both slices have the same validated length.
    Ok(unsafe { kr_kernel_dot_f32(left.as_ptr(), right.as_ptr(), left.len()) })
}

pub struct ModelHandle(*mut c_void);

pub struct AlignedBuffer {
    pointer: *mut c_void,
    bytes: usize,
    alignment: usize,
}

impl AlignedBuffer {
    pub fn new(bytes: usize, alignment: usize) -> Result<Self, &'static str> {
        // SAFETY: allocator validates sizes and returns unique owned memory.
        let pointer = unsafe { kr_memory_alloc_aligned(bytes, alignment) };
        if pointer.is_null() {
            return Err("aligned allocation failed");
        }
        Ok(Self {
            pointer,
            bytes,
            alignment,
        })
    }
    pub fn as_ptr(&self) -> *const c_void {
        self.pointer
    }
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.pointer
    }
    pub fn len_bytes(&self) -> usize {
        self.bytes
    }
    pub fn alignment(&self) -> usize {
        self.alignment
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: AlignedBuffer uniquely owns this pointer.
        unsafe { kr_memory_free_aligned(self.pointer) };
    }
}

impl Drop for ModelHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: ModelHandle uniquely owns the opaque C allocation.
            unsafe { kr_model_destroy(self.0) };
        }
    }
}
