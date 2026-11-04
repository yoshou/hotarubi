# AGENTS.md

Instructions for coding agents working in this repository.

## Project

hotarubi is a toy project: an NPU with its own instruction set on the KV260 that runs SmolVLA and SparseDrive-S. The design is in `docs/design.md`.

## Layout

| Path | Contents |
|---|---|
| `docs/` | Design |
| `rtl/` | SystemVerilog RTL |
| `tb/` | Verilator testbenches in C++ |
| `fpga/` | Vivado Tcl scripts, constraints, device tree |
| `tools/iss/` | Instruction-set simulator: bit-exact arithmetic of each unit and execution of instruction streams |
| `tools/compiler/` | Compiler |
| `tools/runtime/` | Driver and runtime on the board |

## Documents

`docs/design.md` contains only the design: no options, recommendations, rationale, history, work logs or explanatory notes.

## Language

Everything in the repository is written in English.

## C++

- ISO C++23, limited to the features implemented by every compiler below. No compiler extensions, undefined behavior or implementation-defined behavior is relied on.
- Compilers: GCC 14 and Clang 21 on the host; GCC 14 (`g++-14-aarch64-linux-gnu`) for the board. The code builds without warnings, and behaves the same, with each of them.
- Style: [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html), formatted with `clang-format-21` (`.clang-format`).
- Static analysis: `clang-tidy-21` (`.clang-tidy`), run by the `tidy` preset.
- Warnings: `-Wall -Wextra -Wpedantic -Werror`.
- Agents do not write comments in code on their own. Comments are written only where the owner asks for them, and for the closing comments of namespaces and include guards required by the style checks.
- File names are snake_case. Public headers are under `include/hotarubi/`. Test files are named `*_test.cpp`.
- Unit tests use GoogleTest and are added with `hotarubi_add_test` (root `CMakeLists.txt`).
- The runtime for the board is cross-compiled for aarch64 and linked statically.

## SystemVerilog

- Style: [lowRISC SystemVerilog Coding Style Guide](https://github.com/lowRISC/style-guides/blob/master/VerilogCodingStyle.md).
- Lint: Verilator `--lint-only -Wall`.

## Build and test

CMake is the entry point for the C++ code, the Verilator testbenches and the Vivado steps. The list of RTL sources is defined once in CMake.

| Preset | Purpose |
|---|---|
| `gcc-debug`, `gcc-release`, `clang-debug`, `clang-release` | Build and test with each compiler |
| `sanitize` | Tests with AddressSanitizer and UndefinedBehaviorSanitizer |
| `coverage` | Clang source-based coverage; the `coverage-report` target prints the report |
| `tidy` | Build with clang-tidy |
| `aarch64-release` | Cross-compile the runtime for the board |

```
cmake --preset gcc-debug
cmake --build --preset gcc-debug
ctest --preset gcc-debug
```

CI (`.github/workflows/ci.yml`) builds and tests with `gcc-release`, `clang-release` and `sanitize`, reports coverage, checks formatting and runs clang-tidy. Coverage is reported only; it has no threshold.

## Commits

Commit only when the owner asks, on `main`. The message is one line, with no trailers such as Co-Authored-By.
