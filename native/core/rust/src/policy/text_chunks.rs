//! UTF-8-safe, context-preserving text chunk planning.
//!
//! Chunks cover the source exactly once through `content_start..end`.  A
//! following chunk may also include `context_start..content_start` as a prefix
//! for model input.  A trainer must mask that prefix from loss calculation;
//! this crate plans boundaries only and does not claim to be an LLM trainer.

pub const INVALID: usize = usize::MAX;
const MAX_TEXT_BYTES: usize = 2 * 1024 * 1024;
const MAX_CHUNKS: usize = 65_536;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextChunk {
    pub context_start: usize,
    pub content_start: usize,
    pub end: usize,
}

fn before_or_at(values: &[usize], limit: usize) -> Option<usize> {
    let mut result = None;
    for value in values {
        if *value > limit {
            break;
        }
        result = Some(*value);
    }
    result
}

fn after_or_at(values: &[usize], limit: usize, ceiling: usize) -> Option<usize> {
    values
        .iter()
        .copied()
        .find(|value| *value >= limit && *value <= ceiling)
}

fn boundaries(text: &str) -> Vec<usize> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut result = Vec::with_capacity(chars.len() / 8 + 2);
    result.push(0);
    for (index, (offset, character)) in chars.iter().copied().enumerate() {
        let end = offset + character.len_utf8();
        let next = chars.get(index + 1).map(|(_, value)| *value);
        let sentence_end = matches!(character, '.' | '!' | '?' | '…')
            && next.map_or(true, |value| value.is_whitespace() || matches!(value, '"' | '\'' | ')' | ']'));
        if character.is_whitespace() || sentence_end {
            result.push(end);
        }
    }
    result.push(text.len());
    result.sort_unstable();
    result.dedup();
    result
}

pub fn plan(
    text: &str,
    minimum_bytes: usize,
    target_bytes: usize,
    maximum_bytes: usize,
    overlap_bytes: usize,
) -> Option<Vec<TextChunk>> {
    if text.len() > MAX_TEXT_BYTES
        || minimum_bytes == 0
        || target_bytes == 0
        || maximum_bytes < minimum_bytes
        || overlap_bytes >= maximum_bytes
    {
        return None;
    }
    if text.is_empty() {
        return Some(Vec::new());
    }
    let points = boundaries(text);
    let target = target_bytes.clamp(minimum_bytes, maximum_bytes);
    let overlap = overlap_bytes.min(target.saturating_sub(1));
    let mut result = Vec::new();
    let mut content_start = 0;
    while content_start < text.len() {
        if result.len() >= MAX_CHUNKS {
            return None;
        }
        let remaining = text.len() - content_start;
        let ceiling = content_start.saturating_add(maximum_bytes).min(text.len());
        let desired = content_start.saturating_add(target).min(text.len());
        let floor = content_start.saturating_add(minimum_bytes).min(text.len());
        let mut end = if remaining <= maximum_bytes {
            text.len()
        } else {
            before_or_at(&points, desired).filter(|value| *value >= floor)
                .or_else(|| after_or_at(&points, desired, ceiling))
                .or_else(|| before_or_at(&points, ceiling).filter(|value| *value > content_start))
                .unwrap_or(ceiling)
        };
        if end <= content_start {
            end = after_or_at(&points, content_start + 1, text.len()).unwrap_or(text.len());
        }
        let context_limit = content_start.saturating_sub(overlap);
        let context_start = before_or_at(&points, context_limit).unwrap_or(0);
        result.push(TextChunk {
            context_start,
            content_start,
            end,
        });
        content_start = end;
    }
    Some(result)
}

/// Plan into a caller-owned C buffer. Call with a null output first to obtain
/// the required span count. No partial output is written when capacity is low.
///
/// # Safety
/// `text` must point to at least `length` readable bytes. `output` is either
/// null or points to `capacity` writable `TextChunk` values.
#[unsafe(export_name = "kr_rust_policy_plan_text_chunks")]
pub unsafe extern "C" fn kr_rust_plan_text_chunks(
    text: *const u8,
    length: usize,
    minimum_bytes: usize,
    target_bytes: usize,
    maximum_bytes: usize,
    overlap_bytes: usize,
    output: *mut TextChunk,
    capacity: usize,
) -> usize {
    if text.is_null() && length != 0 {
        return INVALID;
    }
    let bytes = if length == 0 { &[] } else { unsafe { core::slice::from_raw_parts(text, length) } };
    let Ok(value) = core::str::from_utf8(bytes) else {
        return INVALID;
    };
    let Some(plan) = plan(value, minimum_bytes, target_bytes, maximum_bytes, overlap_bytes) else {
        return INVALID;
    };
    if output.is_null() || capacity < plan.len() {
        return plan.len();
    }
    unsafe { core::ptr::copy_nonoverlapping(plan.as_ptr(), output, plan.len()) };
    plan.len()
}

#[cfg(test)]
mod tests {
    use super::plan;

    #[test]
    fn content_is_contiguous_and_context_is_utf8_safe() {
        let text = "Первое предложение завершено. Второе предложение продолжает ту же мысль.\n\nТретий абзац завершает пример.";
        let chunks = plan(text, 25, 45, 60, 16).expect("valid plan");
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].content_start, 0);
        assert_eq!(chunks.last().expect("chunk").end, text.len());
        for pair in chunks.windows(2) {
            assert_eq!(pair[0].end, pair[1].content_start);
            assert!(pair[1].context_start <= pair[1].content_start);
            assert!(text.is_char_boundary(pair[1].context_start));
            assert!(text.is_char_boundary(pair[1].content_start));
            assert!(text.is_char_boundary(pair[1].end));
        }
    }

    #[test]
    fn rejects_invalid_overlap_and_invalid_utf8_is_handled_by_ffi() {
        assert!(plan("text", 1, 2, 3, 3).is_none());
    }
}
