cmake_minimum_required(VERSION 3.20)
project(kernelyra_native LANGUAGES CXX Fortran)

add_executable(dataset_probe tools/dataset_probe.cpp)
target_compile_features(dataset_probe PRIVATE cxx_std_17)
set(KERNELYRA_NATIVE_SOURCES
  bridge/cpp/core_abi.cpp
  core/fortran/training_kernels.f90
)
add_library(kernelyra_core SHARED ${KERNELYRA_NATIVE_SOURCES})
target_compile_features(kernelyra_core PRIVATE cxx_std_17)
target_include_directories(kernelyra_core PRIVATE include)
target_compile_definitions(kernelyra_core PRIVATE KR_HAS_FORTRAN_NUMERIC=1)
find_program(KERNELYRA_ZIG_EXECUTABLE zig REQUIRED)
set(KERNELYRA_ZIG_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/memory_zig.obj")
add_custom_command(
  OUTPUT "${KERNELYRA_ZIG_OBJECT}"
  COMMAND "${KERNELYRA_ZIG_EXECUTABLE}" build-obj "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_kernels.zig"
          -O ReleaseFast -target x86_64-windows-gnu -femit-bin=${KERNELYRA_ZIG_OBJECT}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/core/zig/memory_kernels.zig"
  VERBATIM)
set_source_files_properties("${KERNELYRA_ZIG_OBJECT}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
target_sources(kernelyra_core PRIVATE "${KERNELYRA_ZIG_OBJECT}")
target_compile_definitions(kernelyra_core PRIVATE KR_HAS_ZIG_MEMORY=1)
set_target_properties(kernelyra_core PROPERTIES PREFIX "" OUTPUT_NAME "kernelyra_core")
if(MSVC)
  target_compile_options(dataset_probe PRIVATE /W4 /O2)
  target_compile_options(kernelyra_core PRIVATE /W4 /O2 /fp:fast)
else()
  target_compile_options(dataset_probe PRIVATE -Wall -Wextra -Wpedantic -O3)
  target_compile_options(kernelyra_core PRIVATE
    -Wall -Wextra -Wpedantic -O3 -mtune=generic -fno-math-errno -fno-trapping-math -ffp-contract=fast -fopenmp)
  target_link_options(kernelyra_core PRIVATE -fopenmp)
  if(MINGW)
    target_link_options(kernelyra_core PRIVATE -static -static-libgcc -static-libstdc++)
  endif()
endif()

# Fortran owns dense training arithmetic and Zig owns explicit memory operations.
# C++ remains a small ABI and streaming bridge around the low-level core.
