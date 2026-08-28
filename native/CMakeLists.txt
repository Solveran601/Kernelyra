cmake_minimum_required(VERSION 3.20)
project(kernelyra_native LANGUAGES C CXX Fortran)
include(CTest)

add_executable(dataset_probe tools/dataset_probe.cpp)
target_compile_features(dataset_probe PRIVATE cxx_std_17)
set(KERNELYRA_NATIVE_SOURCES
  bridge/cpp/core_abi.cpp
  bridge/cpp/context_policy.cpp
  core/c/execution_guard.c
  core/fortran/numeric_constants.f90
  core/fortran/numeric_precision.f90
  core/fortran/numeric_moments.f90
  core/fortran/activation_softmax.f90
  core/fortran/loss_binary.f90
  core/fortran/loss_multiclass.f90
  core/fortran/loss_regression.f90
  core/fortran/gradient_layout.f90
  core/fortran/workspace_kernels.f90
  core/fortran/vector_kernels.f90
  core/fortran/gradient_kernels.f90
  core/fortran/optimizer_clip.f90
  core/fortran/training_guard.f90
  core/fortran/training_state.f90
  core/fortran/matrix_scores.f90
  core/fortran/multiclass_training.f90
  core/fortran/training_kernels.f90
)
add_library(kernelyra_core SHARED ${KERNELYRA_NATIVE_SOURCES})
target_compile_features(kernelyra_core PRIVATE cxx_std_17)
target_include_directories(kernelyra_core PRIVATE include core/c)
target_compile_definitions(kernelyra_core PRIVATE KR_HAS_FORTRAN_NUMERIC=1)
find_program(KERNELYRA_CARGO_EXECUTABLE cargo REQUIRED)
set(KERNELYRA_RUST_TARGET x86_64-pc-windows-gnu)
set(KERNELYRA_RUST_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo-target")
set(KERNELYRA_RUST_LIBRARY
  "${KERNELYRA_RUST_TARGET_DIR}/${KERNELYRA_RUST_TARGET}/release/libkernelyra_rust_policy.a")
add_custom_command(
  OUTPUT "${KERNELYRA_RUST_LIBRARY}"
  COMMAND "${CMAKE_COMMAND}" -E env "CARGO_TARGET_DIR=${KERNELYRA_RUST_TARGET_DIR}"
          "${KERNELYRA_CARGO_EXECUTABLE}" build --manifest-path
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/Cargo.toml" --release --target ${KERNELYRA_RUST_TARGET}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/Cargo.toml"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/src/lib.rs"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/src/adaptive.rs"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/src/batch_plan.rs"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/src/chunks.rs"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/src/hash.rs"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/src/signature.rs"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/src/sampler.rs"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/src/split.rs"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/src/text_chunks.rs"
  VERBATIM)
add_custom_target(kernelyra_rust_policy DEPENDS "${KERNELYRA_RUST_LIBRARY}")
add_library(kernelyra_rust_policy_static STATIC IMPORTED GLOBAL)
set_target_properties(kernelyra_rust_policy_static PROPERTIES IMPORTED_LOCATION "${KERNELYRA_RUST_LIBRARY}")
add_dependencies(kernelyra_core kernelyra_rust_policy)
target_link_libraries(kernelyra_core PRIVATE kernelyra_rust_policy_static)
if(WIN32)
  target_link_libraries(kernelyra_core PRIVATE ws2_32 userenv bcrypt ntdll)
endif()
target_compile_definitions(kernelyra_core PRIVATE KR_HAS_RUST_POLICY=1)
find_program(KERNELYRA_ZIG_EXECUTABLE zig REQUIRED)
set(KERNELYRA_ZIG_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/memory_zig.obj")
add_custom_command(
  OUTPUT "${KERNELYRA_ZIG_OBJECT}"
  COMMAND "${CMAKE_COMMAND}" -E env
          "ZIG_GLOBAL_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/zig-global-cache"
          "ZIG_LOCAL_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/zig-local-cache"
          "${KERNELYRA_ZIG_EXECUTABLE}" build-obj "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_kernels.zig"
          -O ReleaseFast -target x86_64-windows-gnu -femit-bin=${KERNELYRA_ZIG_OBJECT}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_kernels.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_alignment.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_allocator.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_arena.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_batch.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_budget.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_arithmetic.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_clip.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_finite.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_transfer.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_guard.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_normalize.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_preprocess.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_reduce.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_repair.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_rows.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_shape.zig"
  VERBATIM)
set_source_files_properties("${KERNELYRA_ZIG_OBJECT}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
target_sources(kernelyra_core PRIVATE "${KERNELYRA_ZIG_OBJECT}")
target_compile_definitions(kernelyra_core PRIVATE KR_HAS_ZIG_MEMORY=1)
set_target_properties(kernelyra_core PROPERTIES PREFIX "" OUTPUT_NAME "kernelyra_core")
add_library(kernelyra_c_policy STATIC bindings/c/kernelyra_policy.c)
target_include_directories(kernelyra_c_policy PUBLIC bindings/c/include include)
target_link_libraries(kernelyra_c_policy PUBLIC kernelyra_core)
if(BUILD_TESTING)
  add_executable(kernelyra_native_contract_tests tools/native_contract_tests.cpp)
  target_compile_features(kernelyra_native_contract_tests PRIVATE cxx_std_17)
  target_include_directories(kernelyra_native_contract_tests PRIVATE include bindings/c/include core/c)
  target_link_libraries(kernelyra_native_contract_tests PRIVATE kernelyra_core kernelyra_c_policy)
  add_test(NAME kernelyra_native_contract_tests COMMAND kernelyra_native_contract_tests)
endif()
if(MSVC)
  target_compile_options(dataset_probe PRIVATE /W4 /O2)
  target_compile_options(kernelyra_core PRIVATE /W4 /O2 /fp:fast)
  target_compile_options(kernelyra_c_policy PRIVATE /W4 /O2)
  if(BUILD_TESTING)
    target_compile_options(kernelyra_native_contract_tests PRIVATE /W4 /O2)
  endif()
else()
  target_compile_options(dataset_probe PRIVATE -Wall -Wextra -Wpedantic -O3)
  target_compile_options(kernelyra_core PRIVATE
    -Wall -Wextra -Wpedantic -O3 -mtune=generic -fno-math-errno -fno-trapping-math -ffp-contract=fast -fopenmp)
  target_compile_options(kernelyra_c_policy PRIVATE -Wall -Wextra -Wpedantic -Werror -O3)
  target_link_options(kernelyra_core PRIVATE -fopenmp)
  if(BUILD_TESTING)
    target_compile_options(kernelyra_native_contract_tests PRIVATE -Wall -Wextra -Wpedantic -Werror -O2)
  endif()
  if(MINGW)
    target_link_options(kernelyra_core PRIVATE -static -static-libgcc -static-libstdc++)
  endif()
endif()

# Fortran owns portable dense arithmetic and Zig owns explicit memory operations.
# The C++ bridge also dispatches runtime-checked AVX2/FMA and OpenMP hot paths.
