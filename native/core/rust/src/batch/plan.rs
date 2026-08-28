//! Overflow-safe dimensions for allocation-free native minibatches.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchPlan {
    pub rows: usize,
    pub values: usize,
    pub values_bytes: usize,
}

/// Plan a batch that fits inside the source table. The C++ bridge owns the
/// buffers; this module only proves that the dimensions can be represented.
pub fn plan(source_rows: usize, features: usize, requested_rows: usize) -> Option<BatchPlan> {
    if source_rows == 0 || features == 0 || requested_rows == 0 || requested_rows > source_rows {
        return None;
    }
    let values = requested_rows.checked_mul(features)?;
    let values_bytes = values.checked_mul(core::mem::size_of::<f32>())?;
    Some(BatchPlan {
        rows: requested_rows,
        values,
        values_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::plan;

    #[test]
    fn plan_keeps_batch_inside_source_and_reports_float32_bytes() {
        let batch = plan(128, 17, 32).expect("valid batch");
        assert_eq!(batch.rows, 32);
        assert_eq!(batch.values, 544);
        assert_eq!(batch.values_bytes, 2_176);
    }

    #[test]
    fn plan_rejects_invalid_or_overflowing_dimensions() {
        assert!(plan(0, 4, 1).is_none());
        assert!(plan(4, 4, 5).is_none());
        assert!(plan(usize::MAX, 2, usize::MAX).is_none());
    }
}
