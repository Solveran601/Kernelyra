# Kernelyra native core

The released Windows x64 core has deliberate boundaries. It is not a Python
training loop translated into several languages.

```text
native/
├── core/
│   ├── fortran/     dense training arithmetic: dot, AXPY, gradients, updates
│   ├── rust/        deterministic context split and bounded chunk policies
│   └── zig/         aligned buffers, copy, zero and normalization
├── bridge/cpp/      compact C ABI, bounded CSV streaming and safe fallback
├── bindings/c/      C cursor library for bounded context-safe chunks
├── include/         stable C/C++/Rust-facing ABI header
├── tools/           safe dataset signature probe
└── CMakeLists.txt   reproducible native build
```

Rust, Fortran and Zig are the active low-level engine components. Rust owns
deterministic context-safe train/validation/test assignment and bounded,
variable-size chunk policies; its experimental adaptive extension accepts
explicit memory pressure and algorithm aggression, not guessed PC classes.
C exposes a validated execution-plan contract for CPU/RAM/GPU/thread budgets.
C++ invokes the policy through the stable C ABI, streams batches without loading
a whole dataset, and provides fused checked preprocessing (repair, normalize,
clip) for float32 matrices. The binary classifier keeps its AVX2-capable hot
loop in C++; Fortran owns regression and shared dense numeric primitives.
Native multiclass is still a partial C++ implementation until its equivalent
Fortran kernel is complete. Rust, C and C++ consume the ABI directly; Python is
only the high-level orchestration and optional-framework layer.

End users install a Windows wheel containing `kernelyra_core.dll` and do not
need a compiler. Source contributors need MinGW g++, gfortran, Zig and Rust:

```powershell
kernelyra native build
kernelyra native status --json
```

For a CMake build:

```powershell
cmake -S native -B native/build -G "MinGW Makefiles"
cmake --build native/build --config Release
```

The project intentionally has no handwritten assembly source. CPU-specific
machine code is emitted by Zig, gfortran and the C++ compiler for the actual
target; this avoids shipping one fixed ISA implementation that fails on a
different machine.
