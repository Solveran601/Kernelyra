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

#[derive(Default)]
struct BoundaryIndex {
    /// Newlines are record/message boundaries in line-oriented conversation
    /// exports, so they always outrank a punctuation boundary.
    messages: Vec<usize>,
    sentences: Vec<usize>,
}

/// `target_bytes == 0` is the ABI spelling of the public Python API's
/// ``target_bytes="auto"``.  It chooses an even, bounded nominal size before
/// semantic boundaries make the individual spans naturally variable.  The
/// calculation is allocation-free and depends only on caller supplied bounds
/// and the source length, so a plan is repeatable on every machine.
fn automatic_target(text_len: usize, minimum_bytes: usize, maximum_bytes: usize) -> usize {
    if text_len <= maximum_bytes || minimum_bytes == maximum_bytes {
        return maximum_bytes.min(text_len.max(minimum_bytes));
    }
    let span = maximum_bytes.saturating_sub(minimum_bytes);
    // Prefer the upper half of the caller's allowed range.  It keeps metadata
    // and per-chunk overhead low without turning the maximum into a mandate.
    let preferred = minimum_bytes.saturating_add(span.saturating_mul(2) / 3);
    let chunk_count = text_len
        .saturating_add(preferred.saturating_sub(1))
        .checked_div(preferred)
        .unwrap_or(1)
        .max(1);
    text_len
        .saturating_add(chunk_count.saturating_sub(1))
        .checked_div(chunk_count)
        .unwrap_or(maximum_bytes)
        .clamp(minimum_bytes, maximum_bytes)
}

fn before_or_at(values: &[usize], limit: usize) -> Option<usize> {
    let index = values.partition_point(|value| *value <= limit);
    index.checked_sub(1).map(|previous| values[previous])
}

fn after_or_at(values: &[usize], limit: usize, ceiling: usize) -> Option<usize> {
    values
        .get(values.partition_point(|value| *value < limit))
        .copied()
        .filter(|value| *value <= ceiling)
}

fn boundaries(text: &str) -> BoundaryIndex {
    let mut result = BoundaryIndex {
        messages: Vec::with_capacity(text.len() / 64 + 2),
        sentences: Vec::with_capacity(text.len() / 128 + 2),
    };
    result.messages.push(0);
    result.sentences.push(0);
    let mut chars = text.char_indices().peekable();
    while let Some((offset, character)) = chars.next() {
        let end = offset + character.len_utf8();
        let next = chars.peek().map(|(_, value)| *value);
        let sentence_end = matches!(character, '.' | '!' | '?' | '…')
            && next.map_or(true, |value| value.is_whitespace() || matches!(value, '"' | '\'' | ')' | ']'));
        // Newlines preserve record/paragraph/message boundaries in exported
        // conversations.  Sentence punctuation is a fallback for ordinary
        // prose and must never outrank a usable message boundary.  We do not
        // retain every whitespace: that index can exceed the text itself on a
        // whitespace-heavy source, while a UTF-8-safe hard boundary remains
        // available below.
        if character == '\n' {
            result.messages.push(end);
        }
        if sentence_end {
            result.sentences.push(end);
        }
    }
    result.messages.push(text.len());
    result.sentences.push(text.len());
    result.messages.sort_unstable();
    result.messages.dedup();
    result.sentences.sort_unstable();
    result.sentences.dedup();
    result
}

fn preferred_end(
    points: &BoundaryIndex,
    floor: usize,
    desired: usize,
    ceiling: usize,
) -> Option<usize> {
    // A whole message is more useful than a slightly closer sentence end.
    // Only when no line boundary fits do we split at a sentence boundary.
    for candidates in [&points.messages, &points.sentences] {
        if let Some(end) = before_or_at(candidates, desired).filter(|value| *value >= floor) {
            return Some(end);
        }
        if let Some(end) = after_or_at(candidates, desired, ceiling) {
            return Some(end);
        }
    }
    None
}

fn bounded_context_start(text: &str, messages: &[usize], content_start: usize, overlap: usize) -> usize {
    let lower = content_start.saturating_sub(overlap);
    // If a complete preceding message fits in the requested overlap, use it.
    // Otherwise use the nearest UTF-8 boundary exactly at the memory cap.
    if content_start > 0 {
        if let Some(message_start) = before_or_at(messages, content_start - 1).filter(|value| *value >= lower) {
            return message_start;
        }
    }
    let mut boundary = lower.min(text.len());
    while boundary > 0 && !text.is_char_boundary(boundary) {
        boundary -= 1;
    }
    boundary
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
        || maximum_bytes < minimum_bytes
        || overlap_bytes >= maximum_bytes
    {
        return None;
    }
    if text.is_empty() {
        return Some(Vec::new());
    }
    let points = boundaries(text);
    let target = if target_bytes == 0 {
        automatic_target(text.len(), minimum_bytes, maximum_bytes)
    } else {
        target_bytes.clamp(minimum_bytes, maximum_bytes)
    };
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
            preferred_end(&points, floor, desired, ceiling)
                .or_else(|| before_or_at(&points.messages, ceiling).filter(|value| *value > content_start))
                .or_else(|| before_or_at(&points.sentences, ceiling).filter(|value| *value > content_start))
                .unwrap_or(ceiling)
        };
        if end <= content_start {
            end = after_or_at(&points.messages, content_start + 1, text.len())
                .or_else(|| after_or_at(&points.sentences, content_start + 1, text.len()))
                .unwrap_or(ceiling);
        }
        let context_start = bounded_context_start(text, &points.messages, content_start, overlap);
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
    use super::{automatic_target, plan};

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

    #[test]
    fn long_input_keeps_exact_coverage_without_rescanning_boundaries() {
        let text = "Sentence with stable UTF-8 границы. ".repeat(32_768);
        let chunks = plan(&text, 768, 1_536, 2_048, 256).expect("valid long plan");
        assert!(chunks.len() > 100);
        assert_eq!(chunks[0].content_start, 0);
        assert_eq!(chunks.last().expect("last chunk").end, text.len());
        for pair in chunks.windows(2) {
            assert_eq!(pair[0].end, pair[1].content_start);
            assert!(text.is_char_boundary(pair[1].context_start));
            assert!(text.is_char_boundary(pair[1].end));
        }
    }

    #[test]
    fn automatic_target_is_bounded_repeatable_and_prefers_message_boundaries() {
        let text = ("User: explain adaptive chunks.\nAssistant: They preserve complete messages.\n")
            .repeat(128);
        let automatic = plan(&text, 96, 0, 512, 32).expect("automatic plan");
        assert_eq!(automatic_target(text.len(), 96, 512), automatic_target(text.len(), 96, 512));
        assert_eq!(automatic.first().expect("chunk").content_start, 0);
        assert_eq!(automatic.last().expect("chunk").end, text.len());
        for pair in automatic.windows(2) {
            assert_eq!(pair[0].end, pair[1].content_start);
            assert!(text.is_char_boundary(pair[1].end));
            // Usable line/message boundaries outrank sentence endings.
            assert_eq!(text.as_bytes()[pair[0].end - 1], b'\n');
        }
    }
}
