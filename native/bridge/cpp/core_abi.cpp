#include "kernelyra_core.h"
#include "context_policy.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define KR_X86_GNU_SIMD 1
#else
#define KR_X86_GNU_SIMD 0
#endif

#ifndef KR_HAS_ZIG_MEMORY
#define KR_HAS_ZIG_MEMORY 0
#endif
#ifndef KR_HAS_FORTRAN_NUMERIC
#define KR_HAS_FORTRAN_NUMERIC 0
#endif
#ifndef KR_HAS_RUST_POLICY
#define KR_HAS_RUST_POLICY 0
#endif
extern "C" {
#if KR_HAS_ZIG_MEMORY
void kr_zig_normalize_f32(
    float* data, size_t rows, size_t features, const float* means, const float* stds);
void kr_zig_copy_f32(float* destination, const float* source, size_t values);
void kr_zig_zero_f32(float* destination, size_t values);
uint32_t kr_zig_all_finite_f32(const float* values, size_t count);
void kr_zig_clip_f32(float* values, size_t count, float limit);
void* kr_zig_alloc_aligned(size_t bytes, size_t alignment);
void kr_zig_free_aligned(void* pointer);
#endif
#if KR_HAS_FORTRAN_NUMERIC
void kr_fortran_gradient_f32(
    const float* x, const float* errors, size_t rows, size_t features, float* gradient);
int kr_fortran_all_finite_f32(const float* values, size_t count);
float kr_fortran_l2_norm_f32(const float* values, size_t count);
float kr_fortran_dot_f32(const float* left, const float* right, size_t values);
void kr_fortran_axpy_f32(float* output, const float* row, float scale, size_t values);
void kr_fortran_update_f32(
    float* weights, const float* gradient, float learning_rate, float inverse,
    float decay, size_t values);
void kr_fortran_binary_train_f32(
    const float* x, const float* y, size_t rows, size_t features, float* weights, float* bias,
    float learning_rate, float decay, float* errors, float* gradient, float* loss);
void kr_fortran_regression_train_f32(
    const float* x, const float* y, size_t rows, size_t features, float* weights, float* bias,
    float learning_rate, float decay, float target_mean, float target_std,
    float* errors, float* gradient, float* loss);
#endif
#if KR_HAS_RUST_POLICY
uint64_t kr_rust_policy_mix_u64(uint64_t value);
uint32_t kr_rust_policy_split_for_key(
    uint64_t group_key, uint32_t validation_percent, uint32_t test_percent);
size_t kr_rust_policy_next_chunk_size(
    size_t remaining_records, size_t target_records, size_t minimum_records,
    size_t maximum_records, uint64_t sequence, uint64_t seed);
#endif
}

namespace {

thread_local std::string last_error;
constexpr uint32_t compiled_component_mask =
    (KR_HAS_ZIG_MEMORY ? static_cast<uint32_t>(KR_COMPONENT_ZIG_MEMORY) : 0U) |
    (KR_HAS_FORTRAN_NUMERIC ? static_cast<uint32_t>(KR_COMPONENT_FORTRAN_NUMERIC) : 0U) |
    (KR_HAS_RUST_POLICY ? static_cast<uint32_t>(KR_COMPONENT_RUST_POLICY) : 0U);
// The native kernel is intentionally active by default: Zig owns the explicit
// buffer operations and Fortran owns the dense training arithmetic. A caller
// can still mask components for diagnostics or fallback verification.
std::atomic<uint32_t> enabled_component_mask{compiled_component_mask};

bool component_enabled(uint32_t component) {
  return (enabled_component_mask.load(std::memory_order_relaxed) & component) != 0U;
}

struct Model {
  kr_model_config config{};
  std::vector<float> weights;
  std::vector<float> bias;
  std::vector<float> gradient;
  std::vector<float> bias_gradient;
  std::vector<float> scratch;
  std::vector<float> batch_x;
  std::vector<float> batch_y;
  std::vector<float> parallel_gradient;
  std::vector<float> parallel_bias;
  std::vector<double> parallel_loss;
  std::vector<float> errors;
  uint64_t rng_state = 0;
};

struct NumericCsv {
  size_t rows = 0;
  size_t features = 0;
  std::string target;
  std::vector<std::string> feature_names;
  std::vector<float> means;
  std::vector<float> stds;
  std::vector<float> x;
  std::vector<float> y;
};

struct NumericCsvStream {
  std::filesystem::path path;
  std::ifstream input;
  std::streampos data_start{};
  char delimiter = ',';
  size_t columns = 0;
  std::vector<uint32_t> feature_columns;
  std::vector<int32_t> column_to_feature;
  uint32_t target_column = 0;
  std::vector<float> means;
  std::vector<float> stds;
  uint32_t task = KR_TASK_BINARY;
  std::vector<float> classes;
  uint32_t selected_split = 0;
  uint64_t selected_records = 0;
  uint64_t row_index = 0;
  uint64_t rows_consumed = 0;
};

struct NumericCsvScan {
  uint64_t rows = 0;
  std::array<uint64_t, 3> split_rows{};
  uint32_t target_column = 0;
  std::vector<std::string> columns;
  std::vector<std::string> feature_names;
  std::string target;
  std::vector<float> means;
  std::vector<float> stds;
  std::vector<std::string> target_values;
  bool target_values_overflow = false;
};

int fail(const char* message) {
  last_error = message;
  return 0;
}

std::string trim(std::string value) {
  size_t begin = 0;
  while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) ++begin;
  size_t end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) --end;
  return value.substr(begin, end - begin);
}

std::vector<std::string> split_csv(const std::string& line, char delimiter) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (size_t index = 0; index < line.size(); ++index) {
    const char value = line[index];
    if (value == '"') {
      if (quoted && index + 1 < line.size() && line[index + 1] == '"') {
        field.push_back('"');
        ++index;
      } else {
        quoted = !quoted;
      }
    } else if (value == delimiter && !quoted) {
      fields.push_back(trim(field));
      field.clear();
    } else if (value != '\r') {
      field.push_back(value);
    }
  }
  if (quoted) return {};
  fields.push_back(trim(field));
  return fields;
}

bool parse_numeric_field(const char* text, float& output) {
  const unsigned char* cursor = reinterpret_cast<const unsigned char*>(text);
  while (*cursor != 0 && std::isspace(*cursor)) ++cursor;
  bool negative = false;
  if (*cursor == '+' || *cursor == '-') {
    negative = *cursor == '-';
    ++cursor;
  }
  double mantissa = 0.0;
  int fractional_digits = 0;
  bool decimal = false;
  bool any_digit = false;
  while (*cursor != 0) {
    if (*cursor >= '0' && *cursor <= '9') {
      any_digit = true;
      mantissa = mantissa * 10.0 + static_cast<double>(*cursor - '0');
      if (decimal) ++fractional_digits;
      ++cursor;
      continue;
    }
    if (*cursor == '.' && !decimal) {
      decimal = true;
      ++cursor;
      continue;
    }
    break;
  }
  int explicit_exponent = 0;
  if (*cursor == 'e' || *cursor == 'E') {
    ++cursor;
    bool exponent_negative = false;
    if (*cursor == '+' || *cursor == '-') {
      exponent_negative = *cursor == '-';
      ++cursor;
    }
    bool exponent_digit = false;
    while (*cursor >= '0' && *cursor <= '9') {
      exponent_digit = true;
      explicit_exponent = std::min(1000, explicit_exponent * 10 + static_cast<int>(*cursor - '0'));
      ++cursor;
    }
    if (!exponent_digit) return false;
    if (exponent_negative) explicit_exponent = -explicit_exponent;
  }
  while (*cursor != 0 && std::isspace(*cursor)) ++cursor;
  if (!any_digit || *cursor != 0) return false;
  const int exponent = explicit_exponent - fractional_digits;
  static const std::array<double, 91> powers = [] {
    std::array<double, 91> values{};
    for (int index = -45; index <= 45; ++index) values[static_cast<size_t>(index + 45)] = std::pow(10.0, index);
    return values;
  }();
  if (exponent < -45 || exponent > 45) return false;
  double value = mantissa * powers[static_cast<size_t>(exponent + 45)];
  if (negative) value = -value;
  output = static_cast<float>(value);
  return std::isfinite(output);
}

uint32_t split_for_row(uint64_t index) {
#if KR_HAS_RUST_POLICY
  if (component_enabled(KR_COMPONENT_RUST_POLICY)) {
    const uint32_t split = kr_rust_policy_split_for_key(index, 15U, 15U);
    if (split != std::numeric_limits<uint32_t>::max()) return split;
  }
#endif
  return kernelyra::policy::split_for_context(index, 15U, 15U);
}

bool rewind_stream(NumericCsvStream& stream) {
  stream.input.clear();
  stream.input.seekg(stream.data_start);
  stream.row_index = 0;
  if (!stream.input) return fail("numeric CSV stream could not rewind") != 0;
  return true;
}

bool encode_stream_row(NumericCsvStream& stream, std::string& line, float* x, float& y) {
  if (!line.empty() && line.back() == '\r') line.pop_back();
  size_t column = 0;
  size_t encoded = 0;
  bool target_found = false;
  char* cursor = line.data();
  char* const end = cursor + line.size();
  while (cursor <= end) {
    char* field = cursor;
    while (cursor < end && *cursor != stream.delimiter) {
      if (*cursor == '"') return fail("quoted fields use the general streaming ingestor") != 0;
      ++cursor;
    }
    const bool more = cursor < end;
    if (more) *cursor = '\0';
    if (column >= stream.columns) return fail("numeric CSV stream row is wider than its header") != 0;
    const int32_t feature = stream.column_to_feature[column];
    if (feature >= 0 || column == stream.target_column) {
      float value = 0.0F;
      if (!parse_numeric_field(field, value)) {
        return fail("numeric CSV stream encountered missing or non-numeric data") != 0;
      }
      if (feature >= 0) {
        const size_t index = static_cast<size_t>(feature);
        x[index] = value;
        ++encoded;
      }
      if (column == stream.target_column) {
        target_found = true;
        if (stream.task == KR_TASK_REGRESSION) {
          y = value;
        } else {
          const auto found = std::find(stream.classes.begin(), stream.classes.end(), value);
          if (found == stream.classes.end()) {
            return fail("target class changed after numeric CSV stream inspection") != 0;
          }
          y = static_cast<float>(found - stream.classes.begin());
        }
      }
    }
    ++column;
    if (!more) break;
    ++cursor;
  }
  if (column != stream.columns || encoded != stream.feature_columns.size() || !target_found) {
    return fail("numeric CSV stream row width is inconsistent") != 0;
  }
  return true;
}

NumericCsv* load_numeric_csv(const char* path_utf8, const char* target_utf8, char delimiter) {
  if (path_utf8 == nullptr || *path_utf8 == '\0') {
    fail("CSV path is empty");
    return nullptr;
  }
  if (delimiter == '\0' || delimiter == '"' || delimiter == '\n' || delimiter == '\r') {
    fail("CSV delimiter is invalid");
    return nullptr;
  }
  std::ifstream input(std::filesystem::u8path(path_utf8), std::ios::binary | std::ios::ate);
  if (!input) {
    fail("numeric CSV file could not be opened");
    return nullptr;
  }
  const std::streamsize file_size = input.tellg();
  if (file_size <= 0) {
    fail("numeric CSV is empty");
    return nullptr;
  }
  std::string contents;
  try {
    contents.resize(static_cast<size_t>(file_size));
  } catch (const std::bad_alloc&) {
    fail("numeric CSV file buffer does not fit in memory");
    return nullptr;
  }
  input.seekg(0, std::ios::beg);
  if (!input.read(contents.data(), file_size)) {
    fail("numeric CSV file could not be read completely");
    return nullptr;
  }
  const size_t header_end = contents.find_first_of("\r\n");
  if (header_end == std::string::npos) {
    fail("numeric CSV has no data rows");
    return nullptr;
  }
  std::vector<std::string> columns = split_csv(contents.substr(0, header_end), delimiter);
  if (!columns.empty() && columns[0].size() >= 3 &&
      static_cast<unsigned char>(columns[0][0]) == 0xEF &&
      static_cast<unsigned char>(columns[0][1]) == 0xBB &&
      static_cast<unsigned char>(columns[0][2]) == 0xBF) {
    columns[0].erase(0, 3);
  }
  if (columns.size() < 2) {
    fail("numeric CSV requires a header, features and target");
    return nullptr;
  }
  const std::string requested = target_utf8 == nullptr ? "" : target_utf8;
  size_t target_index = columns.size() - 1;
  if (!requested.empty()) {
    const auto found = std::find(columns.begin(), columns.end(), requested);
    if (found == columns.end()) {
      fail("target column was not found in numeric CSV");
      return nullptr;
    }
    target_index = static_cast<size_t>(found - columns.begin());
  }
  NumericCsv* dataset = new (std::nothrow) NumericCsv();
  if (dataset == nullptr) {
    fail("numeric CSV handle allocation failed");
    return nullptr;
  }
  dataset->target = columns[target_index];
  for (size_t index = 0; index < columns.size(); ++index) {
    if (index != target_index) dataset->feature_names.push_back(columns[index]);
  }
  dataset->features = dataset->feature_names.size();
  std::vector<double> means(dataset->features, 0.0);
  std::vector<double> m2(dataset->features, 0.0);
  try {
    dataset->x.reserve(static_cast<size_t>(file_size / 8));
    dataset->y.reserve(static_cast<size_t>(file_size / std::max<size_t>(16, columns.size() * 8)));
    char* cursor = contents.data() + header_end;
    char* const end = contents.data() + contents.size();
    while (cursor < end && (*cursor == '\r' || *cursor == '\n')) ++cursor;
    size_t column = 0;
    size_t feature_index = 0;
    while (cursor < end) {
      if (*cursor == '\r' || *cursor == '\n') {
        while (cursor < end && (*cursor == '\r' || *cursor == '\n')) ++cursor;
        continue;
      }
      char* field = cursor;
      while (cursor < end && *cursor != delimiter && *cursor != '\r' && *cursor != '\n') {
        if (*cursor == '"') {
          delete dataset;
          fail("quoted CSV fields use the general ingestor instead of the numeric fast path");
          return nullptr;
        }
        ++cursor;
      }
      const char separator = cursor < end ? *cursor : '\0';
      if (cursor < end) *cursor = '\0';
      float value = 0.0F;
      if (!parse_numeric_field(field, value) || column >= columns.size()) {
        delete dataset;
        fail("numeric CSV fast path encountered missing or non-numeric data");
        return nullptr;
      }
      if (column == target_index) {
        dataset->y.push_back(value);
      } else {
        dataset->x.push_back(value);
        const double delta = static_cast<double>(value) - means[feature_index];
        means[feature_index] += delta / static_cast<double>(dataset->rows + 1);
        m2[feature_index] += delta * (static_cast<double>(value) - means[feature_index]);
        ++feature_index;
      }
      ++column;
      if (separator == delimiter) {
        ++cursor;
        continue;
      }
      if (column != columns.size()) {
        delete dataset;
        fail("numeric CSV has an unstable row width");
        return nullptr;
      }
      ++dataset->rows;
      column = 0;
      feature_index = 0;
      if (cursor < end) ++cursor;
      while (cursor < end && (*cursor == '\r' || *cursor == '\n')) ++cursor;
    }
    if (column != 0) {
      if (column != columns.size()) {
        delete dataset;
        fail("numeric CSV final row has an unstable width");
        return nullptr;
      }
      ++dataset->rows;
    }
    if (dataset->rows < 32) {
      delete dataset;
      fail("numeric CSV fast path requires at least 32 rows");
      return nullptr;
    }
    dataset->means.resize(dataset->features);
    dataset->stds.resize(dataset->features);
    for (size_t feature = 0; feature < dataset->features; ++feature) {
      const double variance = m2[feature] / static_cast<double>(dataset->rows);
      const double deviation = std::sqrt(std::max(0.0, variance));
      if (deviation <= 1.0e-12) {
        delete dataset;
        fail("numeric CSV fast path encountered a constant feature");
        return nullptr;
      }
      dataset->means[feature] = static_cast<float>(means[feature]);
      dataset->stds[feature] = static_cast<float>(deviation);
    }
#if KR_HAS_ZIG_MEMORY
    if (component_enabled(KR_COMPONENT_ZIG_MEMORY)) {
      kr_zig_normalize_f32(
          dataset->x.data(), dataset->rows, dataset->features,
          dataset->means.data(), dataset->stds.data());
    } else
#endif
    {
#if defined(_OPENMP)
#pragma omp parallel for if(dataset->rows * dataset->features >= 1048576U) schedule(static)
#endif
      for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(dataset->rows); ++row) {
        for (size_t feature = 0; feature < dataset->features; ++feature) {
          float& value = dataset->x[static_cast<size_t>(row) * dataset->features + feature];
          value = (value - dataset->means[feature]) / dataset->stds[feature];
        }
      }
    }
  } catch (const std::bad_alloc&) {
    delete dataset;
    fail("numeric CSV does not fit in the available memory budget");
    return nullptr;
  }
  return dataset;
}

NumericCsvScan* scan_numeric_csv(const char* path_utf8, const char* target_utf8, char delimiter) {
  if (path_utf8 == nullptr || *path_utf8 == '\0') {
    fail("CSV scan path is empty");
    return nullptr;
  }
  std::ifstream input(std::filesystem::u8path(path_utf8), std::ios::binary);
  std::string header;
  if (!input || !std::getline(input, header)) {
    fail("numeric CSV scan could not read its header");
    return nullptr;
  }
  std::vector<std::string> columns = split_csv(header, delimiter);
  if (!columns.empty() && columns[0].size() >= 3 &&
      static_cast<unsigned char>(columns[0][0]) == 0xEF &&
      static_cast<unsigned char>(columns[0][1]) == 0xBB &&
      static_cast<unsigned char>(columns[0][2]) == 0xBF) {
    columns[0].erase(0, 3);
  }
  if (columns.size() < 2) {
    fail("numeric CSV scan requires a header, features and target");
    return nullptr;
  }
  const std::string requested = target_utf8 == nullptr ? "" : target_utf8;
  size_t target_column = columns.size() - 1;
  if (!requested.empty()) {
    const auto found = std::find(columns.begin(), columns.end(), requested);
    if (found == columns.end()) {
      fail("target column was not found during numeric CSV scan");
      return nullptr;
    }
    target_column = static_cast<size_t>(found - columns.begin());
  }
  NumericCsvScan* scan = new (std::nothrow) NumericCsvScan();
  if (scan == nullptr) {
    fail("numeric CSV scan allocation failed");
    return nullptr;
  }
  scan->columns = columns;
  scan->target_column = static_cast<uint32_t>(target_column);
  scan->target = columns[target_column];
  for (size_t column = 0; column < columns.size(); ++column) {
    if (column != target_column) scan->feature_names.push_back(columns[column]);
  }
  const size_t features = scan->feature_names.size();
  std::vector<double> means(features, 0.0);
  std::vector<double> m2(features, 0.0);
  uint64_t train_rows = 0;
  std::string line;
  try {
    while (std::getline(input, line)) {
      if (trim(line).empty()) continue;
      std::vector<std::string> fields = split_csv(line, delimiter);
      if (fields.size() != columns.size()) {
        delete scan;
        fail("numeric CSV scan encountered an unstable row width");
        return nullptr;
      }
      const uint32_t split = split_for_row(scan->rows);
      ++scan->split_rows[split];
      const bool train = split == 0U;
      size_t feature = 0;
      for (size_t column = 0; column < fields.size(); ++column) {
        float value = 0.0F;
        if (!parse_numeric_field(fields[column].c_str(), value)) {
          delete scan;
          fail("numeric CSV scan encountered missing or non-numeric data");
          return nullptr;
        }
        if (column == target_column) {
          if (!scan->target_values_overflow &&
              std::find(scan->target_values.begin(), scan->target_values.end(), fields[column]) ==
                  scan->target_values.end()) {
            if (scan->target_values.size() < 65) {
              scan->target_values.push_back(fields[column]);
            } else {
              scan->target_values_overflow = true;
            }
          }
        } else {
          if (train) {
            const double delta = static_cast<double>(value) - means[feature];
            means[feature] += delta / static_cast<double>(train_rows + 1);
            m2[feature] += delta * (static_cast<double>(value) - means[feature]);
          }
          ++feature;
        }
      }
      if (train) ++train_rows;
      ++scan->rows;
    }
    if (scan->rows < 32 || train_rows < 8) {
      delete scan;
      fail("numeric CSV scan requires at least 32 rows and 8 training rows");
      return nullptr;
    }
    std::sort(scan->target_values.begin(), scan->target_values.end());
    scan->means.resize(features);
    scan->stds.resize(features);
    for (size_t feature = 0; feature < features; ++feature) {
      const double variance = m2[feature] / static_cast<double>(train_rows);
      const double deviation = std::sqrt(std::max(0.0, variance));
      if (deviation <= 1.0e-12) {
        delete scan;
        fail("numeric CSV scan encountered a constant feature");
        return nullptr;
      }
      scan->means[feature] = static_cast<float>(means[feature]);
      scan->stds[feature] = static_cast<float>(deviation);
    }
  } catch (const std::bad_alloc&) {
    delete scan;
    fail("numeric CSV scan metadata does not fit in memory");
    return nullptr;
  }
  return scan;
}

uint64_t next_random(uint64_t& state) {
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  return state * 2685821657736338717ULL;
}

float random_weight(uint64_t& state) {
  const uint64_t value = next_random(state);
  const float unit = static_cast<float>((value >> 40) & 0xFFFFFFU) / 16777216.0F;
  return (unit - 0.5F) * 0.2F;
}

float sigmoid(float value) {
  value = std::clamp(value, -30.0F, 30.0F);
  return 1.0F / (1.0F + std::exp(-value));
}

float dot_scalar(const float* row, const float* weights, size_t features) {
  float sum = 0.0F;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
  for (size_t feature = 0; feature < features; ++feature) {
    sum += row[feature] * weights[feature];
  }
  return sum;
}

void add_scaled_scalar(float* output, const float* row, float scale, size_t features) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
  for (size_t feature = 0; feature < features; ++feature) output[feature] += row[feature] * scale;
}

void update_scalar(
    float* weights, const float* gradient, float learning_rate, float inverse,
    float decay, size_t features) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
  for (size_t feature = 0; feature < features; ++feature) {
    weights[feature] -= learning_rate * (gradient[feature] * inverse + decay * weights[feature]);
  }
}

#if KR_X86_GNU_SIMD
bool runtime_avx2_fma() {
  static const bool available = [] {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  }();
  return available;
}

__attribute__((target("avx2,fma")))
float dot_avx2(const float* row, const float* weights, size_t features) {
  __m256 sum = _mm256_setzero_ps();
  size_t feature = 0;
  for (; feature + 8 <= features; feature += 8) {
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(row + feature), _mm256_loadu_ps(weights + feature), sum);
  }
  const __m128 low = _mm256_castps256_ps128(sum);
  const __m128 high = _mm256_extractf128_ps(sum, 1);
  __m128 combined = _mm_add_ps(low, high);
  combined = _mm_hadd_ps(combined, combined);
  combined = _mm_hadd_ps(combined, combined);
  float result = _mm_cvtss_f32(combined);
  for (; feature < features; ++feature) result += row[feature] * weights[feature];
  return result;
}

__attribute__((target("avx2,fma")))
void add_scaled_avx2(float* output, const float* row, float scale, size_t features) {
  const __m256 factor = _mm256_set1_ps(scale);
  size_t feature = 0;
  for (; feature + 8 <= features; feature += 8) {
    const __m256 value = _mm256_fmadd_ps(_mm256_loadu_ps(row + feature), factor, _mm256_loadu_ps(output + feature));
    _mm256_storeu_ps(output + feature, value);
  }
  for (; feature < features; ++feature) output[feature] += row[feature] * scale;
}

__attribute__((target("avx2,fma")))
void update_avx2(
    float* weights, const float* gradient, float learning_rate, float inverse,
    float decay, size_t features) {
  const __m256 rate = _mm256_set1_ps(learning_rate);
  const __m256 scale = _mm256_set1_ps(inverse);
  const __m256 regularization = _mm256_set1_ps(decay);
  size_t feature = 0;
  for (; feature + 8 <= features; feature += 8) {
    const __m256 current = _mm256_loadu_ps(weights + feature);
    const __m256 change = _mm256_fmadd_ps(
        _mm256_loadu_ps(gradient + feature), scale, _mm256_mul_ps(regularization, current));
    _mm256_storeu_ps(weights + feature, _mm256_fnmadd_ps(rate, change, current));
  }
  for (; feature < features; ++feature) {
    weights[feature] -= learning_rate * (gradient[feature] * inverse + decay * weights[feature]);
  }
}
#else
bool runtime_avx2_fma() { return false; }
#endif

float dot(const float* row, const float* weights, size_t features) {
#if KR_X86_GNU_SIMD
  if (runtime_avx2_fma()) {
    return dot_avx2(row, weights, features);
  }
#endif
#if KR_HAS_FORTRAN_NUMERIC
  if (component_enabled(KR_COMPONENT_FORTRAN_NUMERIC)) {
    return kr_fortran_dot_f32(row, weights, features);
  }
#endif
  return dot_scalar(row, weights, features);
}

void add_scaled(float* output, const float* row, float scale, size_t features) {
#if KR_X86_GNU_SIMD
  if (runtime_avx2_fma()) {
    add_scaled_avx2(output, row, scale, features);
    return;
  }
#endif
#if KR_HAS_FORTRAN_NUMERIC
  if (component_enabled(KR_COMPONENT_FORTRAN_NUMERIC)) {
    kr_fortran_axpy_f32(output, row, scale, features);
    return;
  }
#endif
  add_scaled_scalar(output, row, scale, features);
}

void update_weights(
    float* weights, const float* gradient, float learning_rate, float inverse,
    float decay, size_t features) {
#if KR_HAS_FORTRAN_NUMERIC
  if (component_enabled(KR_COMPONENT_FORTRAN_NUMERIC)) {
    kr_fortran_update_f32(weights, gradient, learning_rate, inverse, decay, features);
    return;
  }
#endif
#if KR_X86_GNU_SIMD
  if (runtime_avx2_fma()) {
    update_avx2(weights, gradient, learning_rate, inverse, decay, features);
    return;
  }
#endif
  update_scalar(weights, gradient, learning_rate, inverse, decay, features);
}

bool valid_model(const Model* model) {
  return model != nullptr && model->config.abi_version == KR_ABI_VERSION &&
         model->config.features > 0;
}

bool values_are_finite(const float* values, size_t count) {
  if (values == nullptr) return false;
#if KR_HAS_ZIG_MEMORY
  if (component_enabled(KR_COMPONENT_ZIG_MEMORY)) return kr_zig_all_finite_f32(values, count) != 0U;
#endif
#if KR_HAS_FORTRAN_NUMERIC
  if (component_enabled(KR_COMPONENT_FORTRAN_NUMERIC)) return kr_fortran_all_finite_f32(values, count) != 0;
#endif
  for (size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) return false;
  }
  return true;
}

bool model_is_finite(const Model& model) {
  return values_are_finite(model.weights.data(), model.weights.size()) &&
         values_are_finite(model.bias.data(), model.bias.size());
}

int configured_threads(const Model& model, size_t rows) {
  const size_t requested = std::max<size_t>(1, model.config.threads);
  return static_cast<int>(std::min(requested, rows));
}

#if defined(_OPENMP)
int train_binary_parallel(
    Model& model, const float* x, const float* y, size_t rows, float* loss) {
  const size_t features = model.config.features;
  const int threads = configured_threads(model, rows);
  // Do not call a Fortran scalar kernel once per row here.  A local C++ AVX2
  // accumulator keeps the entire hot loop in one compiled region; Fortran is
  // still used by the single-thread and bulk numerical paths.
  model.parallel_gradient.assign(static_cast<size_t>(threads) * features, 0.0F);
  model.parallel_bias.assign(threads, 0.0F);
  model.parallel_loss.assign(threads, 0.0);
#pragma omp parallel num_threads(threads)
  {
    const int thread = omp_get_thread_num();
    float* gradient = model.parallel_gradient.data() + static_cast<size_t>(thread) * features;
    float bias_gradient = 0.0F;
    double total_loss = 0.0;
#pragma omp for schedule(static)
    for (ptrdiff_t row_index = 0; row_index < static_cast<ptrdiff_t>(rows); ++row_index) {
      const float* row = x + static_cast<size_t>(row_index) * features;
      const float probability = sigmoid(dot(row, model.weights.data(), features) + model.bias[0]);
      const float error = probability - y[row_index];
      const float bounded = std::clamp(probability, 1.0e-7F, 1.0F - 1.0e-7F);
      total_loss -= static_cast<double>(y[row_index]) * std::log(bounded) +
                    static_cast<double>(1.0F - y[row_index]) * std::log(1.0F - bounded);
      bias_gradient += error;
      add_scaled(gradient, row, error, features);
    }
    model.parallel_bias[thread] = bias_gradient;
    model.parallel_loss[thread] = total_loss;
  }
  std::fill(model.gradient.begin(), model.gradient.end(), 0.0F);
  for (int thread = 0; thread < threads; ++thread) {
    add_scaled(model.gradient.data(),
               model.parallel_gradient.data() + static_cast<size_t>(thread) * features,
               1.0F, features);
  }
  const float inverse = 1.0F / static_cast<float>(rows);
  update_weights(model.weights.data(), model.gradient.data(), model.config.learning_rate,
                 inverse, model.config.weight_decay, features);
  float bias_gradient = 0.0F;
  double total_loss = 0.0;
  for (int thread = 0; thread < threads; ++thread) {
    bias_gradient += model.parallel_bias[thread];
    total_loss += model.parallel_loss[thread];
  }
  model.bias[0] -= model.config.learning_rate * bias_gradient * inverse;
  *loss = static_cast<float>(total_loss / static_cast<double>(rows));
  return std::isfinite(*loss) ? 1 : fail("parallel binary loss became non-finite");
}

int train_regression_parallel(
    Model& model, const float* x, const float* y, size_t rows, float* loss) {
  const size_t features = model.config.features;
  const int threads = configured_threads(model, rows);
  const float target_std = std::abs(model.config.target_std) > 1.0e-12F ? model.config.target_std : 1.0F;
#if KR_HAS_FORTRAN_NUMERIC
  if (component_enabled(KR_COMPONENT_FORTRAN_NUMERIC)) {
    model.errors.resize(rows);
    double total_loss = 0.0;
    float bias_gradient = 0.0F;
#pragma omp parallel for num_threads(threads) reduction(+:total_loss,bias_gradient) schedule(static)
    for (ptrdiff_t row_index = 0; row_index < static_cast<ptrdiff_t>(rows); ++row_index) {
      const float* row = x + static_cast<size_t>(row_index) * features;
      const float target = (y[row_index] - model.config.target_mean) / target_std;
      const float error = dot(row, model.weights.data(), features) + model.bias[0] - target;
      model.errors[static_cast<size_t>(row_index)] = 2.0F * error;
      total_loss += static_cast<double>(error) * error;
      bias_gradient += 2.0F * error;
    }
    kr_fortran_gradient_f32(x, model.errors.data(), rows, features, model.gradient.data());
    const float inverse = 1.0F / static_cast<float>(rows);
    update_weights(model.weights.data(), model.gradient.data(), model.config.learning_rate,
                   inverse, model.config.weight_decay, features);
    model.bias[0] -= model.config.learning_rate * bias_gradient * inverse;
    *loss = static_cast<float>(total_loss / static_cast<double>(rows));
    return std::isfinite(*loss) ? 1 : fail("Fortran regression loss became non-finite");
  }
#endif
  model.parallel_gradient.assign(static_cast<size_t>(threads) * features, 0.0F);
  model.parallel_bias.assign(threads, 0.0F);
  model.parallel_loss.assign(threads, 0.0);
#pragma omp parallel num_threads(threads)
  {
    const int thread = omp_get_thread_num();
    float* gradient = model.parallel_gradient.data() + static_cast<size_t>(thread) * features;
    float bias_gradient = 0.0F;
    double total_loss = 0.0;
#pragma omp for schedule(static)
    for (ptrdiff_t row_index = 0; row_index < static_cast<ptrdiff_t>(rows); ++row_index) {
      const float* row = x + static_cast<size_t>(row_index) * features;
      const float target = (y[row_index] - model.config.target_mean) / target_std;
      const float error = dot(row, model.weights.data(), features) + model.bias[0] - target;
      total_loss += static_cast<double>(error) * error;
      bias_gradient += 2.0F * error;
      add_scaled(gradient, row, 2.0F * error, features);
    }
    model.parallel_bias[thread] = bias_gradient;
    model.parallel_loss[thread] = total_loss;
  }
  std::fill(model.gradient.begin(), model.gradient.end(), 0.0F);
  for (int thread = 0; thread < threads; ++thread) {
    add_scaled(model.gradient.data(),
               model.parallel_gradient.data() + static_cast<size_t>(thread) * features,
               1.0F, features);
  }
  const float inverse = 1.0F / static_cast<float>(rows);
  update_weights(model.weights.data(), model.gradient.data(), model.config.learning_rate,
                 inverse, model.config.weight_decay, features);
  float bias_gradient = 0.0F;
  double total_loss = 0.0;
  for (int thread = 0; thread < threads; ++thread) {
    bias_gradient += model.parallel_bias[thread];
    total_loss += model.parallel_loss[thread];
  }
  model.bias[0] -= model.config.learning_rate * bias_gradient * inverse;
  *loss = static_cast<float>(total_loss / static_cast<double>(rows));
  return std::isfinite(*loss) ? 1 : fail("parallel regression loss became non-finite");
}
#endif

int train_binary(Model& model, const float* x, const float* y, size_t rows, float* loss) {
  const size_t features = model.config.features;
#if defined(_OPENMP)
  if (model.config.threads > 1 && rows * features >= 1048576U) {
    return train_binary_parallel(model, x, y, rows, loss);
  }
#endif
#if KR_HAS_FORTRAN_NUMERIC
  if (component_enabled(KR_COMPONENT_FORTRAN_NUMERIC)) {
    model.errors.resize(rows);
    kr_fortran_binary_train_f32(
        x, y, rows, features, model.weights.data(), model.bias.data(), model.config.learning_rate,
        model.config.weight_decay, model.errors.data(), model.gradient.data(), loss);
    return std::isfinite(*loss) ? 1 : fail("Fortran binary loss became non-finite");
  }
#endif
  std::fill(model.gradient.begin(), model.gradient.end(), 0.0F);
  float bias_gradient = 0.0F;
  double total_loss = 0.0;
  for (size_t row_index = 0; row_index < rows; ++row_index) {
    const float* row = x + row_index * features;
    const float probability = sigmoid(dot(row, model.weights.data(), features) + model.bias[0]);
    const float error = probability - y[row_index];
    const float bounded = std::clamp(probability, 1.0e-7F, 1.0F - 1.0e-7F);
    total_loss -= static_cast<double>(y[row_index]) * std::log(bounded) +
                  static_cast<double>(1.0F - y[row_index]) * std::log(1.0F - bounded);
    bias_gradient += error;
    add_scaled(model.gradient.data(), row, error, features);
  }
  const float inverse = 1.0F / static_cast<float>(rows);
  update_weights(model.weights.data(), model.gradient.data(), model.config.learning_rate,
                 inverse, model.config.weight_decay, features);
  model.bias[0] -= model.config.learning_rate * bias_gradient * inverse;
  *loss = static_cast<float>(total_loss / static_cast<double>(rows));
  return std::isfinite(*loss) ? 1 : fail("binary loss became non-finite");
}

int train_regression(Model& model, const float* x, const float* y, size_t rows, float* loss) {
  const size_t features = model.config.features;
#if KR_HAS_FORTRAN_NUMERIC
  if (component_enabled(KR_COMPONENT_FORTRAN_NUMERIC)) {
    model.errors.resize(rows);
    kr_fortran_regression_train_f32(
        x, y, rows, features, model.weights.data(), model.bias.data(), model.config.learning_rate,
        model.config.weight_decay, model.config.target_mean, model.config.target_std,
        model.errors.data(), model.gradient.data(), loss);
    return std::isfinite(*loss) ? 1 : fail("Fortran regression loss became non-finite");
  }
#endif
#if defined(_OPENMP)
  if (model.config.threads > 1 && rows * features >= 1048576U) {
    return train_regression_parallel(model, x, y, rows, loss);
  }
#endif
  std::fill(model.gradient.begin(), model.gradient.end(), 0.0F);
  float bias_gradient = 0.0F;
  double total_loss = 0.0;
  const float target_std = std::abs(model.config.target_std) > 1.0e-12F ? model.config.target_std : 1.0F;
  for (size_t row_index = 0; row_index < rows; ++row_index) {
    const float* row = x + row_index * features;
    const float target = (y[row_index] - model.config.target_mean) / target_std;
    const float error = dot(row, model.weights.data(), features) + model.bias[0] - target;
    total_loss += static_cast<double>(error) * error;
    bias_gradient += 2.0F * error;
    add_scaled(model.gradient.data(), row, 2.0F * error, features);
  }
  const float inverse = 1.0F / static_cast<float>(rows);
  update_weights(model.weights.data(), model.gradient.data(), model.config.learning_rate,
                 inverse, model.config.weight_decay, features);
  model.bias[0] -= model.config.learning_rate * bias_gradient * inverse;
  *loss = static_cast<float>(total_loss / static_cast<double>(rows));
  return std::isfinite(*loss) ? 1 : fail("regression loss became non-finite");
}

#if defined(_OPENMP)
int train_multiclass_parallel(Model& model, const float* x, const float* y, size_t rows, float* loss) {
  const size_t features = model.config.features;
  const size_t classes = model.config.classes;
  const size_t weight_values = model.weights.size();
  const int threads = configured_threads(model, rows);
  model.parallel_gradient.assign(static_cast<size_t>(threads) * weight_values, 0.0F);
  model.parallel_bias.assign(static_cast<size_t>(threads) * classes, 0.0F);
  model.parallel_loss.assign(static_cast<size_t>(threads), 0.0);
  std::atomic<bool> invalid_target{false};
#pragma omp parallel num_threads(threads)
  {
    const int thread = omp_get_thread_num();
    float* gradient = model.parallel_gradient.data() + static_cast<size_t>(thread) * weight_values;
    float* bias_gradient = model.parallel_bias.data() + static_cast<size_t>(thread) * classes;
    std::vector<float> scratch(classes);
    double total_loss = 0.0;
#pragma omp for schedule(static)
    for (ptrdiff_t row_index = 0; row_index < static_cast<ptrdiff_t>(rows); ++row_index) {
      const float* row = x + static_cast<size_t>(row_index) * features;
      const size_t truth = static_cast<size_t>(std::max(0.0F, y[row_index]));
      if (truth >= classes) {
        invalid_target.store(true, std::memory_order_relaxed);
        continue;
      }
      float maximum = -std::numeric_limits<float>::infinity();
      for (size_t category = 0; category < classes; ++category) {
        float value = model.bias[category];
        for (size_t feature = 0; feature < features; ++feature) {
          value += row[feature] * model.weights[feature * classes + category];
        }
        scratch[category] = value;
        maximum = std::max(maximum, value);
      }
      float denominator = 0.0F;
      for (size_t category = 0; category < classes; ++category) {
        scratch[category] = std::exp(scratch[category] - maximum);
        denominator += scratch[category];
      }
      const float truth_probability = scratch[truth] / denominator;
      total_loss -= std::log(std::max(1.0e-7F, truth_probability));
      for (size_t category = 0; category < classes; ++category) {
        const float error = scratch[category] / denominator - (category == truth ? 1.0F : 0.0F);
        bias_gradient[category] += error;
        for (size_t feature = 0; feature < features; ++feature) {
          gradient[feature * classes + category] += row[feature] * error;
        }
      }
    }
    model.parallel_loss[thread] = total_loss;
  }
  if (invalid_target.load(std::memory_order_relaxed)) {
    return fail("multiclass target is outside configured class range");
  }
  std::fill(model.gradient.begin(), model.gradient.end(), 0.0F);
  std::fill(model.bias_gradient.begin(), model.bias_gradient.end(), 0.0F);
  double total_loss = 0.0;
  for (int thread = 0; thread < threads; ++thread) {
    add_scaled(model.gradient.data(),
               model.parallel_gradient.data() + static_cast<size_t>(thread) * weight_values,
               1.0F, weight_values);
    for (size_t category = 0; category < classes; ++category) {
      model.bias_gradient[category] += model.parallel_bias[static_cast<size_t>(thread) * classes + category];
    }
    total_loss += model.parallel_loss[thread];
  }
  const float inverse = 1.0F / static_cast<float>(rows);
  update_weights(model.weights.data(), model.gradient.data(), model.config.learning_rate,
                 inverse, model.config.weight_decay, weight_values);
  for (size_t category = 0; category < classes; ++category) {
    model.bias[category] -= model.config.learning_rate * model.bias_gradient[category] * inverse;
  }
  *loss = static_cast<float>(total_loss / static_cast<double>(rows));
  return std::isfinite(*loss) ? 1 : fail("parallel multiclass loss became non-finite");
}
#endif

int train_multiclass(Model& model, const float* x, const float* y, size_t rows, float* loss) {
  const size_t features = model.config.features;
  const size_t classes = model.config.classes;
#if defined(_OPENMP)
  constexpr size_t max_parallel_gradient_values = 8U * 1024U * 1024U;
  const size_t threads = static_cast<size_t>(configured_threads(model, rows));
  if (model.config.threads > 1 && rows * features >= 1048576U &&
      model.weights.size() <= max_parallel_gradient_values / std::max<size_t>(1U, threads)) {
    return train_multiclass_parallel(model, x, y, rows, loss);
  }
#endif
  std::fill(model.gradient.begin(), model.gradient.end(), 0.0F);
  std::fill(model.bias_gradient.begin(), model.bias_gradient.end(), 0.0F);
  double total_loss = 0.0;
  for (size_t row_index = 0; row_index < rows; ++row_index) {
    const float* row = x + row_index * features;
    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t category = 0; category < classes; ++category) {
      float value = model.bias[category];
      for (size_t feature = 0; feature < features; ++feature) {
        value += row[feature] * model.weights[feature * classes + category];
      }
      model.scratch[category] = value;
      maximum = std::max(maximum, value);
    }
    float denominator = 0.0F;
    for (size_t category = 0; category < classes; ++category) {
      model.scratch[category] = std::exp(model.scratch[category] - maximum);
      denominator += model.scratch[category];
    }
    const size_t truth = static_cast<size_t>(std::max(0.0F, y[row_index]));
    if (truth >= classes) return fail("multiclass target is outside configured class range");
    const float truth_probability = model.scratch[truth] / denominator;
    total_loss -= std::log(std::max(1.0e-7F, truth_probability));
    for (size_t category = 0; category < classes; ++category) {
      const float error = model.scratch[category] / denominator - (category == truth ? 1.0F : 0.0F);
      model.bias_gradient[category] += error;
      for (size_t feature = 0; feature < features; ++feature) {
        model.gradient[feature * classes + category] += row[feature] * error;
      }
    }
  }
  const float inverse = 1.0F / static_cast<float>(rows);
  for (size_t index = 0; index < model.weights.size(); ++index) {
    model.weights[index] -= model.config.learning_rate *
        (model.gradient[index] * inverse + model.config.weight_decay * model.weights[index]);
  }
  for (size_t category = 0; category < classes; ++category) {
    model.bias[category] -= model.config.learning_rate * model.bias_gradient[category] * inverse;
  }
  *loss = static_cast<float>(total_loss / static_cast<double>(rows));
  return std::isfinite(*loss) ? 1 : fail("multiclass loss became non-finite");
}

}  // namespace

extern "C" {

const char* kr_core_version(void) { return "kernelyra-native/3.0"; }

const char* kr_core_features(void) {
#if KR_X86_GNU_SIMD
  if (runtime_avx2_fma()) {
#if defined(_OPENMP)
    return "polyglot-native+avx2-fma+openmp-runtime";
#else
    return "polyglot-native+avx2-fma-runtime";
#endif
  }
  return "sse2-autovectorized";
#elif defined(__SSE2__) || defined(_M_X64)
  return "sse2-autovectorized";
#elif defined(__aarch64__) || defined(_M_ARM64)
  return "arm64-neon-autovectorized";
#else
  return "portable-scalar";
#endif
}

const char* kr_core_components(void) {
  if (compiled_component_mask == KR_COMPONENT_ALL) {
    return "cpp-abi+rust-policy+fortran-training+zig-memory";
  }
  if (compiled_component_mask == 0U) return "cpp-fallback";
  return "cpp+partial-native-components";
}

uint64_t kr_rust_mix_u64(uint64_t value) {
#if KR_HAS_RUST_POLICY
  if (component_enabled(KR_COMPONENT_RUST_POLICY)) return kr_rust_policy_mix_u64(value);
#endif
  return kernelyra::policy::mix_u64(value);
}

uint32_t kr_rust_split_for_key(
    uint64_t group_key, uint32_t validation_percent, uint32_t test_percent) {
  if (validation_percent > 95U || test_percent > 95U || validation_percent + test_percent > 95U) {
    return std::numeric_limits<uint32_t>::max();
  }
#if KR_HAS_RUST_POLICY
  if (component_enabled(KR_COMPONENT_RUST_POLICY)) {
    return kr_rust_policy_split_for_key(group_key, validation_percent, test_percent);
  }
#endif
  return kernelyra::policy::split_for_context(group_key, validation_percent, test_percent);
}

size_t kr_rust_next_chunk_size(
    size_t remaining_records, size_t target_records, size_t minimum_records,
    size_t maximum_records, uint64_t sequence, uint64_t seed) {
  if (remaining_records == 0U || target_records == 0U || minimum_records == 0U ||
      maximum_records < minimum_records) return 0U;
#if KR_HAS_RUST_POLICY
  if (component_enabled(KR_COMPONENT_RUST_POLICY)) {
    return kr_rust_policy_next_chunk_size(
        remaining_records, target_records, minimum_records, maximum_records, sequence, seed);
  }
#endif
  return kernelyra::policy::next_chunk_size(
      remaining_records, target_records, minimum_records, maximum_records, sequence, seed);
}

uint32_t kr_core_component_mask(void) { return compiled_component_mask; }

uint32_t kr_core_enabled_component_mask(void) {
  return enabled_component_mask.load(std::memory_order_relaxed);
}

uint32_t kr_core_set_component_mask(uint32_t mask) {
  return enabled_component_mask.exchange(mask & compiled_component_mask, std::memory_order_relaxed);
}

void kr_memory_normalize_f32(
    float* data, size_t rows, size_t features, const float* means, const float* stds) {
  if (data == nullptr || means == nullptr || stds == nullptr) return;
#if KR_HAS_ZIG_MEMORY
  if (component_enabled(KR_COMPONENT_ZIG_MEMORY)) {
    kr_zig_normalize_f32(data, rows, features, means, stds);
    return;
  }
#endif
#if defined(_OPENMP)
#pragma omp parallel for if(rows * features >= 1048576U) schedule(static)
#endif
  for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(rows); ++row) {
    for (size_t feature = 0; feature < features; ++feature) {
      const size_t index = static_cast<size_t>(row) * features + feature;
      data[index] = (data[index] - means[feature]) / stds[feature];
    }
  }
}

void kr_memory_copy_f32(float* destination, const float* source, size_t values) {
  if (destination == nullptr || source == nullptr || values == 0) return;
#if KR_HAS_ZIG_MEMORY
  if (component_enabled(KR_COMPONENT_ZIG_MEMORY)) {
    kr_zig_copy_f32(destination, source, values);
    return;
  }
#endif
  std::copy(source, source + values, destination);
}

void kr_memory_zero_f32(float* destination, size_t values) {
  if (destination == nullptr || values == 0) return;
#if KR_HAS_ZIG_MEMORY
  if (component_enabled(KR_COMPONENT_ZIG_MEMORY)) {
    kr_zig_zero_f32(destination, values);
    return;
  }
#endif
  std::fill(destination, destination + values, 0.0F);
}

uint32_t kr_values_all_finite_f32(const float* values, size_t count) {
  return values_are_finite(values, count) ? 1U : 0U;
}

float kr_values_l2_norm_f32(const float* values, size_t count) {
  if (values == nullptr || !values_are_finite(values, count)) return std::numeric_limits<float>::quiet_NaN();
#if KR_HAS_FORTRAN_NUMERIC
  if (component_enabled(KR_COMPONENT_FORTRAN_NUMERIC)) return kr_fortran_l2_norm_f32(values, count);
#endif
  double total = 0.0;
  for (size_t index = 0; index < count; ++index) {
    total += static_cast<double>(values[index]) * static_cast<double>(values[index]);
  }
  return static_cast<float>(std::sqrt(total));
}

void kr_values_clip_f32(float* values, size_t count, float limit) {
  if (values == nullptr || !std::isfinite(limit) || limit <= 0.0F) return;
#if KR_HAS_ZIG_MEMORY
  if (component_enabled(KR_COMPONENT_ZIG_MEMORY)) {
    kr_zig_clip_f32(values, count, limit);
    return;
  }
#endif
  for (size_t index = 0; index < count; ++index) {
    values[index] = std::clamp(values[index], -limit, limit);
  }
}

void kr_numeric_gradient_f32(
    const float* x, const float* errors, size_t rows, size_t features, float* gradient) {
  if (x == nullptr || errors == nullptr || gradient == nullptr) return;
#if KR_HAS_FORTRAN_NUMERIC
  if (component_enabled(KR_COMPONENT_FORTRAN_NUMERIC)) {
    kr_fortran_gradient_f32(x, errors, rows, features, gradient);
    return;
  }
#endif
  kr_memory_zero_f32(gradient, features);
  for (size_t row = 0; row < rows; ++row) {
    add_scaled(gradient, x + row * features, errors[row], features);
  }
}

float kr_kernel_dot_f32(const float* left, const float* right, size_t values) {
  if (left == nullptr || right == nullptr) return 0.0F;
  return dot(left, right, values);
}

const char* kr_last_error(void) { return last_error.c_str(); }

void* kr_memory_alloc_aligned(size_t bytes, size_t alignment) {
#if KR_HAS_ZIG_MEMORY
  return kr_zig_alloc_aligned(bytes, alignment);
#else
  (void)bytes;
  (void)alignment;
  return nullptr;
#endif
}

void kr_memory_free_aligned(void* pointer) {
#if KR_HAS_ZIG_MEMORY
  kr_zig_free_aligned(pointer);
#else
  (void)pointer;
#endif
}

void* kr_model_create(const kr_model_config* config) {
  last_error.clear();
  if (config == nullptr || config->abi_version != KR_ABI_VERSION) {
    fail("unsupported native ABI version");
    return nullptr;
  }
  if (config->features == 0 || config->features > 1000000U) {
    fail("feature count is outside the supported range");
    return nullptr;
  }
  if (config->task > KR_TASK_REGRESSION) {
    fail("unknown task type");
    return nullptr;
  }
  const size_t classes = config->task == KR_TASK_MULTICLASS ? config->classes : 1U;
  if (classes == 0 || classes > 65536U) {
    fail("class count is outside the supported range");
    return nullptr;
  }
  Model* model = new (std::nothrow) Model();
  if (model == nullptr) {
    fail("native model allocation failed");
    return nullptr;
  }
  try {
    model->config = *config;
    model->weights.resize(static_cast<size_t>(config->features) * classes);
    model->bias.assign(classes, 0.0F);
    model->gradient.resize(model->weights.size());
    model->bias_gradient.resize(classes);
    model->scratch.resize(classes);
    uint64_t state = config->seed == 0 ? 0x9E3779B97F4A7C15ULL : config->seed;
    for (float& weight : model->weights) weight = random_weight(state);
    model->rng_state = state;
  } catch (const std::bad_alloc&) {
    delete model;
    fail("native model buffers do not fit in memory");
    return nullptr;
  }
  return model;
}

void kr_model_destroy(void* handle) { delete static_cast<Model*>(handle); }

int kr_model_train_step(void* handle, const float* x, const float* y, size_t rows, float* loss) {
  last_error.clear();
  Model* model = static_cast<Model*>(handle);
  if (!valid_model(model) || x == nullptr || y == nullptr || loss == nullptr || rows == 0) {
    return fail("invalid native train_step arguments");
  }
  int outcome = model->config.task == KR_TASK_BINARY ? train_binary(*model, x, y, rows, loss) :
      model->config.task == KR_TASK_MULTICLASS ? train_multiclass(*model, x, y, rows, loss) :
      train_regression(*model, x, y, rows, loss);
  if (outcome != 0 && (!std::isfinite(*loss) || !model_is_finite(*model))) {
    return fail("native update rejected: non-finite parameters or loss");
  }
  return outcome;
}

int kr_model_train_random_step(
    void* handle, const float* x, const float* y, size_t rows, size_t batch_size, float* loss) {
  last_error.clear();
  Model* model = static_cast<Model*>(handle);
  if (!valid_model(model) || x == nullptr || y == nullptr || loss == nullptr ||
      rows == 0 || batch_size == 0 || batch_size > 1048576U) {
    return fail("invalid native random train_step arguments");
  }
  try {
    model->batch_x.resize(batch_size * model->config.features);
    model->batch_y.resize(batch_size);
  } catch (const std::bad_alloc&) {
    return fail("native random batch does not fit in memory");
  }
  const size_t features = model->config.features;
  for (size_t sample = 0; sample < batch_size; ++sample) {
    const size_t selected = static_cast<size_t>(next_random(model->rng_state) % rows);
    const float* source = x + selected * features;
    std::copy(source, source + features, model->batch_x.data() + sample * features);
    model->batch_y[sample] = y[selected];
  }
  return kr_model_train_step(handle, model->batch_x.data(), model->batch_y.data(), batch_size, loss);
}

int kr_model_train_random_steps(
    void* handle,
    const float* x,
    const float* y,
    size_t rows,
    size_t batch_size,
    size_t steps,
    float* loss) {
  if (steps == 0 || steps > 1000000U) return fail("native random train step count is outside bounds");
  for (size_t step = 0; step < steps; ++step) {
    if (!kr_model_train_random_step(handle, x, y, rows, batch_size, loss)) return 0;
  }
  return 1;
}

int kr_model_predict(const void* handle, const float* x, size_t rows, float* output, size_t output_values) {
  last_error.clear();
  const Model* model = static_cast<const Model*>(handle);
  if (!valid_model(model) || x == nullptr || output == nullptr) return fail("invalid native predict arguments");
  const size_t features = model->config.features;
  const size_t classes = model->config.task == KR_TASK_MULTICLASS ? model->config.classes : 1U;
  if (output_values < rows * classes) return fail("native predict output buffer is too small");
  for (size_t row_index = 0; row_index < rows; ++row_index) {
    const float* row = x + row_index * features;
    if (model->config.task == KR_TASK_MULTICLASS) {
      float maximum = -std::numeric_limits<float>::infinity();
      for (size_t category = 0; category < classes; ++category) {
        float value = model->bias[category];
        for (size_t feature = 0; feature < features; ++feature) {
          value += row[feature] * model->weights[feature * classes + category];
        }
        output[row_index * classes + category] = value;
        maximum = std::max(maximum, value);
      }
      float denominator = 0.0F;
      for (size_t category = 0; category < classes; ++category) {
        float& value = output[row_index * classes + category];
        value = std::exp(value - maximum);
        denominator += value;
      }
      for (size_t category = 0; category < classes; ++category) {
        output[row_index * classes + category] /= denominator;
      }
    } else {
      float value = dot(row, model->weights.data(), features) + model->bias[0];
      output[row_index] = model->config.task == KR_TASK_BINARY
          ? sigmoid(value)
          : value * model->config.target_std + model->config.target_mean;
    }
  }
  return 1;
}

size_t kr_model_weight_count(const void* handle) {
  const Model* model = static_cast<const Model*>(handle);
  return valid_model(model) ? model->weights.size() : 0;
}

size_t kr_model_bias_count(const void* handle) {
  const Model* model = static_cast<const Model*>(handle);
  return valid_model(model) ? model->bias.size() : 0;
}

int kr_model_export(const void* handle, float* weights, size_t weight_count, float* bias, size_t bias_count) {
  const Model* model = static_cast<const Model*>(handle);
  if (!valid_model(model) || weights == nullptr || bias == nullptr ||
      weight_count != model->weights.size() || bias_count != model->bias.size()) {
    return fail("native export buffer shape mismatch");
  }
  std::copy(model->weights.begin(), model->weights.end(), weights);
  std::copy(model->bias.begin(), model->bias.end(), bias);
  return 1;
}

int kr_model_import(void* handle, const float* weights, size_t weight_count, const float* bias, size_t bias_count) {
  Model* model = static_cast<Model*>(handle);
  if (!valid_model(model) || weights == nullptr || bias == nullptr ||
      weight_count != model->weights.size() || bias_count != model->bias.size()) {
    return fail("native import buffer shape mismatch");
  }
  std::copy(weights, weights + weight_count, model->weights.begin());
  std::copy(bias, bias + bias_count, model->bias.begin());
  return 1;
}

void* kr_csv_load_numeric(const char* path_utf8, const char* target_utf8, char delimiter) {
  last_error.clear();
  return load_numeric_csv(path_utf8, target_utf8, delimiter);
}

void kr_csv_destroy(void* handle) { delete static_cast<NumericCsv*>(handle); }

size_t kr_csv_rows(const void* handle) {
  const NumericCsv* dataset = static_cast<const NumericCsv*>(handle);
  return dataset == nullptr ? 0 : dataset->rows;
}

size_t kr_csv_features(const void* handle) {
  const NumericCsv* dataset = static_cast<const NumericCsv*>(handle);
  return dataset == nullptr ? 0 : dataset->features;
}

const char* kr_csv_target_name(const void* handle) {
  const NumericCsv* dataset = static_cast<const NumericCsv*>(handle);
  return dataset == nullptr ? "" : dataset->target.c_str();
}

const char* kr_csv_feature_name(const void* handle, size_t index) {
  const NumericCsv* dataset = static_cast<const NumericCsv*>(handle);
  if (dataset == nullptr || index >= dataset->feature_names.size()) return "";
  return dataset->feature_names[index].c_str();
}

float kr_csv_feature_mean(const void* handle, size_t index) {
  const NumericCsv* dataset = static_cast<const NumericCsv*>(handle);
  return dataset == nullptr || index >= dataset->means.size() ? 0.0F : dataset->means[index];
}

float kr_csv_feature_std(const void* handle, size_t index) {
  const NumericCsv* dataset = static_cast<const NumericCsv*>(handle);
  return dataset == nullptr || index >= dataset->stds.size() ? 0.0F : dataset->stds[index];
}

int kr_csv_copy(const void* handle, float* x, size_t x_values, float* y, size_t y_values) {
  const NumericCsv* dataset = static_cast<const NumericCsv*>(handle);
  if (dataset == nullptr || x == nullptr || y == nullptr ||
      x_values != dataset->x.size() || y_values != dataset->y.size()) {
    return fail("numeric CSV output buffer shape mismatch");
  }
  std::copy(dataset->x.begin(), dataset->x.end(), x);
  std::copy(dataset->y.begin(), dataset->y.end(), y);
  return 1;
}

void* kr_csv_stream_open(
    const char* path_utf8,
    char delimiter,
    const uint32_t* feature_columns,
    size_t feature_count,
    uint32_t target_column,
    const float* means,
    const float* stds,
    uint32_t task,
    const float* classes,
    size_t class_count,
    uint32_t selected_split,
    uint64_t selected_records) {
  last_error.clear();
  if (path_utf8 == nullptr || *path_utf8 == '\0' || feature_columns == nullptr ||
      feature_count == 0 || means == nullptr || stds == nullptr || task > KR_TASK_REGRESSION ||
      selected_split > 2 || selected_records == 0) {
    fail("numeric CSV stream configuration is invalid");
    return nullptr;
  }
  if (task != KR_TASK_REGRESSION && (classes == nullptr || class_count < 2)) {
    fail("numeric CSV classification stream requires target classes");
    return nullptr;
  }
  NumericCsvStream* stream = new (std::nothrow) NumericCsvStream();
  if (stream == nullptr) {
    fail("numeric CSV stream allocation failed");
    return nullptr;
  }
  stream->path = std::filesystem::u8path(path_utf8);
  stream->input.open(stream->path, std::ios::binary);
  std::string header;
  if (!stream->input || !std::getline(stream->input, header)) {
    delete stream;
    fail("numeric CSV stream could not read its header");
    return nullptr;
  }
  const std::vector<std::string> columns = split_csv(header, delimiter);
  if (columns.size() < 2 || target_column >= columns.size()) {
    delete stream;
    fail("numeric CSV stream header does not match its specification");
    return nullptr;
  }
  stream->delimiter = delimiter;
  stream->columns = columns.size();
  stream->target_column = target_column;
  stream->task = task;
  stream->selected_split = selected_split;
  stream->selected_records = selected_records;
  stream->data_start = stream->input.tellg();
  stream->column_to_feature.assign(columns.size(), -1);
  try {
    stream->feature_columns.assign(feature_columns, feature_columns + feature_count);
    stream->means.assign(means, means + feature_count);
    stream->stds.assign(stds, stds + feature_count);
    if (classes != nullptr) stream->classes.assign(classes, classes + class_count);
    for (size_t feature = 0; feature < feature_count; ++feature) {
      const uint32_t column = stream->feature_columns[feature];
      if (column >= columns.size() || column == target_column ||
          stream->column_to_feature[column] >= 0 || !std::isfinite(stream->means[feature]) ||
          !std::isfinite(stream->stds[feature]) || stream->stds[feature] <= 1.0e-12F) {
        delete stream;
        fail("numeric CSV stream feature mapping is invalid");
        return nullptr;
      }
      stream->column_to_feature[column] = static_cast<int32_t>(feature);
    }
  } catch (const std::bad_alloc&) {
    delete stream;
    fail("numeric CSV stream metadata does not fit in memory");
    return nullptr;
  }
  return stream;
}

void kr_csv_stream_destroy(void* handle) { delete static_cast<NumericCsvStream*>(handle); }

int kr_csv_stream_next_batch(
    void* handle, size_t batch_size, float* x, size_t x_values, float* y, size_t y_values) {
  NumericCsvStream* stream = static_cast<NumericCsvStream*>(handle);
  if (stream == nullptr || batch_size == 0 || x == nullptr || y == nullptr ||
      x_values != batch_size * stream->feature_columns.size() || y_values != batch_size) {
    return fail("numeric CSV stream output buffer shape mismatch");
  }
  size_t produced = 0;
  std::string line;
  while (produced < batch_size) {
    if (!std::getline(stream->input, line)) {
      if (!rewind_stream(*stream)) return 0;
      continue;
    }
    const uint64_t current = stream->row_index++;
    if (split_for_row(current) != stream->selected_split) continue;
    float target = 0.0F;
    if (!encode_stream_row(
            *stream, line, x + produced * stream->feature_columns.size(), target)) {
      return 0;
    }
    y[produced] = target;
    ++produced;
  }
  stream->rows_consumed += produced;
  kr_memory_normalize_f32(
      x, produced, stream->feature_columns.size(), stream->means.data(), stream->stds.data());
  return 1;
}

uint64_t kr_csv_stream_rows_consumed(const void* handle) {
  const NumericCsvStream* stream = static_cast<const NumericCsvStream*>(handle);
  return stream == nullptr ? 0 : stream->rows_consumed;
}

int kr_csv_stream_restore(void* handle, uint64_t rows_consumed) {
  NumericCsvStream* stream = static_cast<NumericCsvStream*>(handle);
  if (stream == nullptr || !rewind_stream(*stream)) return fail("numeric CSV stream is invalid");
  const uint64_t offset = rows_consumed % stream->selected_records;
  uint64_t skipped = 0;
  std::string line;
  while (skipped < offset) {
    if (!std::getline(stream->input, line)) {
      return fail("numeric CSV stream ended before its checkpoint cursor");
    }
    if (split_for_row(stream->row_index++) == stream->selected_split) ++skipped;
  }
  stream->rows_consumed = rows_consumed;
  return 1;
}

void* kr_csv_scan_numeric(const char* path_utf8, const char* target_utf8, char delimiter) {
  last_error.clear();
  return scan_numeric_csv(path_utf8, target_utf8, delimiter);
}

void kr_csv_scan_destroy(void* handle) { delete static_cast<NumericCsvScan*>(handle); }

uint64_t kr_csv_scan_rows(const void* handle) {
  const NumericCsvScan* scan = static_cast<const NumericCsvScan*>(handle);
  return scan == nullptr ? 0 : scan->rows;
}

uint64_t kr_csv_scan_split_rows(const void* handle, uint32_t split) {
  const NumericCsvScan* scan = static_cast<const NumericCsvScan*>(handle);
  return scan == nullptr || split >= scan->split_rows.size() ? 0 : scan->split_rows[split];
}

size_t kr_csv_scan_features(const void* handle) {
  const NumericCsvScan* scan = static_cast<const NumericCsvScan*>(handle);
  return scan == nullptr ? 0 : scan->feature_names.size();
}

size_t kr_csv_scan_columns(const void* handle) {
  const NumericCsvScan* scan = static_cast<const NumericCsvScan*>(handle);
  return scan == nullptr ? 0 : scan->columns.size();
}

const char* kr_csv_scan_column_name(const void* handle, size_t index) {
  const NumericCsvScan* scan = static_cast<const NumericCsvScan*>(handle);
  return scan == nullptr || index >= scan->columns.size() ? "" : scan->columns[index].c_str();
}

uint32_t kr_csv_scan_target_column(const void* handle) {
  const NumericCsvScan* scan = static_cast<const NumericCsvScan*>(handle);
  return scan == nullptr ? 0 : scan->target_column;
}

const char* kr_csv_scan_target_name(const void* handle) {
  const NumericCsvScan* scan = static_cast<const NumericCsvScan*>(handle);
  return scan == nullptr ? "" : scan->target.c_str();
}

const char* kr_csv_scan_feature_name(const void* handle, size_t index) {
  const NumericCsvScan* scan = static_cast<const NumericCsvScan*>(handle);
  return scan == nullptr || index >= scan->feature_names.size() ? "" : scan->feature_names[index].c_str();
}

float kr_csv_scan_feature_mean(const void* handle, size_t index) {
  const NumericCsvScan* scan = static_cast<const NumericCsvScan*>(handle);
  return scan == nullptr || index >= scan->means.size() ? 0.0F : scan->means[index];
}

float kr_csv_scan_feature_std(const void* handle, size_t index) {
  const NumericCsvScan* scan = static_cast<const NumericCsvScan*>(handle);
  return scan == nullptr || index >= scan->stds.size() ? 0.0F : scan->stds[index];
}

size_t kr_csv_scan_target_values(const void* handle) {
  const NumericCsvScan* scan = static_cast<const NumericCsvScan*>(handle);
  return scan == nullptr ? 0 : scan->target_values.size() + (scan->target_values_overflow ? 1 : 0);
}

const char* kr_csv_scan_target_value(const void* handle, size_t index) {
  const NumericCsvScan* scan = static_cast<const NumericCsvScan*>(handle);
  if (scan == nullptr || index >= scan->target_values.size()) return "";
  return scan->target_values[index].c_str();
}

}  // extern "C"
