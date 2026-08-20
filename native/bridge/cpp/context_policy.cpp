#include "context_policy.hpp"

#include <algorithm>
#include <limits>

namespace kernelyra::policy {
namespace {
constexpr uint64_t golden_ratio = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t mix_a = 0xBF58476D1CE4E5B9ULL;
constexpr uint64_t mix_b = 0x94D049BB133111EBULL;
}  // namespace

uint64_t mix_u64(uint64_t value) {
  value += golden_ratio;
  value = (value ^ (value >> 30U)) * mix_a;
  value = (value ^ (value >> 27U)) * mix_b;
  return value ^ (value >> 31U);
}

uint32_t split_for_context(uint64_t context_key, uint32_t validation_percent, uint32_t test_percent) {
  if (validation_percent > 95U || test_percent > 95U || validation_percent + test_percent > 95U) {
    return std::numeric_limits<uint32_t>::max();
  }
  const uint32_t bucket = static_cast<uint32_t>(mix_u64(context_key) % 100ULL);
  if (bucket < validation_percent) return 1U;
  if (bucket < validation_percent + test_percent) return 2U;
  return 0U;
}

size_t next_chunk_size(
    size_t remaining_records, size_t target_records, size_t minimum_records, size_t maximum_records,
    uint64_t sequence, uint64_t seed) {
  if (remaining_records == 0U || target_records == 0U || minimum_records == 0U ||
      maximum_records < minimum_records) return 0U;
  const size_t lower = std::clamp(target_records * 3U / 4U, minimum_records, maximum_records);
  const size_t upper = std::clamp(target_records * 5U / 4U, lower, maximum_records);
  const size_t proposed = lower + static_cast<size_t>(mix_u64(seed ^ sequence) % (upper - lower + 1U));
  return std::max(std::min(proposed, remaining_records), std::min(remaining_records, minimum_records));
}

}  // namespace kernelyra::policy
