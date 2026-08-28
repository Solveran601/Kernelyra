# Kernelyra native core

The released Windows x64 core has deliberate boundaries. It is not a Python
training loop translated into several languages.

```text
native/
├── core/
│   ├── c/           checked size arithmetic and CPU-memory execution contracts
│   ├── cpp/         ABI/model lifetime bridge and deterministic fallback policy
│   ├── fortran/     numerics/ + training/: moments, losses, optimizers and update kernels
│   ├── rust/        policy/ + batch/: deterministic splits, planners and samplers
│   └── zig/         memory/ + pipeline/: aligned arenas, gathering, transforms and guards
├── bindings/        language-facing adapters that call the stable ABI, never duplicate a kernel
│   └── c/           C cursor library for bounded context-safe chunks
├── include/         stable C/C++/Rust-facing ABI header
├── tools/           safe dataset signature probe
└── CMakeLists.txt   reproducible native build
```

All five directories in `core/` are compiled into the released native binary.
Fortran owns
the active binary, multiclass and regression update paths, together with stable
moments, softmax, cross-entropy, dense scores, gradients and opt-in L2 gradient
clipping. Its files are divided by numerical responsibility so a loss,
activation or optimizer can be tested and replaced without duplicating rules.
Zig owns aligned allocation, row gathering for random batches, transfer,
non-finite repair, finite-value guards, normalization and bounded reductions.
Rust owns deterministic context-safe train/validation/test
assignment, bounded variable-size chunk policies and UTF-8 text-span planning.
Its adaptive extension accepts explicit memory pressure and algorithm
aggression, not guessed PC classes.
C exposes the overflow-checked execution and workspace contracts before any
other core allocates. C++ is the deliberately small ABI, CSV-streaming and model-lifetime bridge; it
does not own the normal numerical update path. It retains scalar/AVX fallbacks
only for diagnostics and builds without the Fortran component. Rust, C and C++
consume the ABI directly; Python is only the high-level orchestration and
optional-framework layer.

The text-span planner is an experimental **preprocessing** primitive. It makes
UTF-8-safe spans whose content covers the original source contiguously and
whose later spans include a bounded prefix of earlier text. Kernelyra v5 does
not yet include a language-model trainer, tokenizer integration, loss masking,
or group-exclusive evaluation for text; the prefix must not be treated as a
claim that LLM fine-tuning is already available.

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
