cmake_minimum_required(VERSION 3.20)
project(kernelyra_native LANGUAGES C CXX Fortran)
include(CTest)

add_executable(dataset_probe tools/dataset_probe.cpp)
target_compile_features(dataset_probe PRIVATE cxx_std_17)
set(KERNELYRA_NATIVE_SOURCES
  core/cpp/abi/core_abi.cpp
  core/cpp/policy/context_policy.cpp
  core/c/contract/checked_arithmetic.c
  core/c/contract/matrix_contract.c
  core/c/contract/batch_contract.c
  core/fortran/numerics/numeric_constants.f90
  core/fortran/numerics/numeric_precision.f90
  core/fortran/numerics/numeric_moments.f90
  core/fortran/numerics/activation_softmax.f90
  core/fortran/numerics/loss_binary.f90
  core/fortran/numerics/loss_multiclass.f90
  core/fortran/numerics/loss_regression.f90
  core/fortran/training/gradient_layout.f90
  core/fortran/training/workspace_kernels.f90
  core/fortran/numerics/vector_kernels.f90
  core/fortran/training/gradient_kernels.f90
  core/fortran/numerics/optimizer_clip.f90
  core/fortran/training/training_guard.f90
  core/fortran/training/training_state.f90
  core/fortran/training/matrix_scores.f90
  core/fortran/training/multiclass_training.f90
  core/fortran/training/training_kernels.f90
)
add_library(kernelyra_core SHARED ${KERNELYRA_NATIVE_SOURCES})
target_compile_features(kernelyra_core PRIVATE cxx_std_17)
target_include_directories(kernelyra_core PRIVATE include core/c/include core/c/contract)
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
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/src/policy/adaptive.rs"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/src/batch/plan.rs"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/src/policy/chunks.rs"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/src/policy/hash.rs"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/src/policy/signature.rs"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/src/batch/sampler.rs"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/src/policy/split.rs"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/rust/src/policy/text_chunks.rs"
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
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory/alignment.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory/allocator.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory/arena.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/pipeline/batch.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory/budget.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory/arithmetic.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/pipeline/clip.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory/finite.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory/transfer.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/pipeline/guard.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/pipeline/normalize.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/pipeline/preprocess.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/pipeline/reduce.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/pipeline/repair.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/pipeline/rows.zig"
          "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory/shape.zig"
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
  target_include_directories(kernelyra_native_contract_tests PRIVATE include bindings/c/include core/c/include)
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
