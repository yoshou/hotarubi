# hotarubi

This is a toy project.

A neural processing unit with its own instruction set, implemented in the programmable logic of the AMD Kria KV260 (Zynq UltraScale+ XCK26), that runs SmolVLA and SparseDrive-S.

## Layout

| Path | Contents |
|---|---|
| `docs/design.md` | Design |
| `AGENTS.md` | Instructions for coding agents, including the coding conventions |
| `CMakeLists.txt`, `CMakePresets.json` | Build entry point and presets |
| `.github/workflows/ci.yml` | CI |

## Build and test

Requirements: CMake 3.25 or later, Ninja, GCC 14, LLVM 21 (`clang-21`, `clang-format-21`, `clang-tidy-21`, `llvm-21`), and GoogleTest. The runtime for the board also needs `g++-14-aarch64-linux-gnu`.

```
cmake --preset gcc-debug
cmake --build --preset gcc-debug
ctest --preset gcc-debug
```

The presets are listed in `AGENTS.md`.

## License

MIT. See [LICENSE](LICENSE).
