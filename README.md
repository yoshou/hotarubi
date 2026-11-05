# hotarubi

This is a toy project.

A neural processing unit with its own instruction set, implemented in the programmable logic of the AMD Kria KV260 (Zynq UltraScale+ XCK26), that runs SmolVLA and SparseDrive-S.

## Layout

| Path | Contents |
|---|---|
| `docs/design.md` | Design |
| `rtl/` | SystemVerilog RTL |
| `tb/` | Verilator testbenches in C++ |
| `fpga/` | Vivado flow: Tcl script, Verilog wrapper, device tree overlay, shell description |
| `include/hotarubi/` | Public C++ headers |
| `tools/iss/` | Instruction-set simulator |
| `tools/compiler/` | Compiler |
| `tools/runtime/` | Driver, runtime and the program that runs on the board |
| `scripts/` | Quantization, board inputs and FP32 references, comparison of board outputs |
| `AGENTS.md` | Instructions for coding agents, including the coding conventions |
| `CMakeLists.txt`, `CMakePresets.json` | Build entry point and presets |
| `.github/workflows/ci.yml` | CI |

## Build and test

Requirements: CMake 3.25 or later, Ninja, GCC 14, LLVM 21 (`clang-21`, `clang-format-21`, `clang-tidy-21`, `llvm-21`), Verilator, and GoogleTest. The runtime for the board also needs `g++-14-aarch64-linux-gnu`.

```
cmake --preset gcc-debug
cmake --build --preset gcc-debug
ctest --preset gcc-debug
```

The presets are listed in `AGENTS.md`.

## Evaluation on the board

These steps quantize and compile SmolVLA and SparseDrive-S, run them on the KV260, and compare the outputs with the FP32 models. Commands run on the host from the repository root; `kv260` stands for the ssh host name of the board.

Additional requirements on the host: Python 3.12 with `venv`, `ffmpeg`, `curl`, `rsync`, and Vivado 2025.2 with the KV260 SOM board files.

### 1. Python environment

```
python3 -m venv build/.venv
build/.venv/bin/pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
build/.venv/bin/pip install "lerobot[smolvla]==0.6.1" pyarrow torch==2.11.0 torchvision==0.26.0
```

### 2. Models and data

```
build/.venv/bin/hf download lerobot/smolvla_base --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6
build/.venv/bin/hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct
build/.venv/bin/hf download lerobot/svla_so100_pickplace --repo-type dataset \
    --revision 728583b5eaf9e739a7f119e2def466fa1d552402 --local-dir build/data/svla_so100_pickplace
mkdir -p build/sparsedrive
curl -L -o build/sparsedrive/sparsedrive_stage2.pth \
    https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2.pth
```

- The scripts read the models from the Hugging Face cache without network access. lerobot takes the SmolVLM2 configuration and processor from its `main` revision; the evaluation used `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- nuScenes: download the v1.0-mini dataset and the CAN bus expansion from https://www.nuscenes.org/nuscenes (registration required) and extract both into `build/data/nuscenes`, so that it holds `v1.0-mini/`, `samples/`, `sweeps/`, `maps/` and `can_bus/`.

### 3. Quantization

```
build/.venv/bin/python scripts/quantize_smolvla.py
build/.venv/bin/python scripts/quantize_sparsedrive.py
```

They write `build/quant/smolvla.safetensors` and `build/quant/sparsedrive.safetensors`.

### 4. Board setup (once)

The board runs Ubuntu 22.04 for Kria; its default boot arguments already reserve the CMA area and enable `generic-uio`. Build the u-dma-buf driver on the board, load it at boot with one 256 MB buffer below 4 GB, give the `plugdev` group (which includes the default `ubuntu` user) access to the devices, and make the firmware directory writable:

```
ssh kv260
git clone --depth 1 https://github.com/ikwzm/udmabuf.git
cd udmabuf && make KERNEL_SRC_DIR=/lib/modules/$(uname -r)/build
sudo mkdir -p /lib/modules/$(uname -r)/extra
sudo cp u-dma-buf.ko /lib/modules/$(uname -r)/extra/ && sudo depmod -a
echo u-dma-buf | sudo tee /etc/modules-load.d/u-dma-buf.conf
echo "options u-dma-buf udmabuf0=268435456 dma_mask_bit=32" | sudo tee /etc/modprobe.d/u-dma-buf.conf
printf 'SUBSYSTEM=="uio", MODE="0660", GROUP="plugdev"\nKERNEL=="udmabuf*", MODE="0660", GROUP="plugdev"\n' |
    sudo tee /etc/udev/rules.d/90-npu.rules
sudo udevadm control --reload-rules && sudo modprobe u-dma-buf && sudo udevadm trigger
sudo mkdir -p /lib/firmware/xilinx/npu && sudo chown $USER: /lib/firmware/xilinx/npu
cat /sys/class/u-dma-buf/udmabuf0/phys_addr
```

The last command prints the physical address of the buffer, which the compiler takes as `--base`. It is `0x38700000` on the board used here; replace it below if it differs.

### 5. Compilation

```
cmake --preset gcc-release
cmake --build --preset gcc-release
build/gcc-release/tools/compiler/hotarubi_compile smolvla --quant build/quant/smolvla.safetensors \
    --model ~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6/model.safetensors \
    --base 0x38700000 --lang-tokens 12 --cameras 2 --steps 10 --output build/compiled/smolvla
build/gcc-release/tools/compiler/hotarubi_compile sparsedrive --quant build/quant/sparsedrive.safetensors \
    --checkpoint build/sparsedrive/sparsedrive_stage2.pth --base 0x38700000 --output build/compiled/sparsedrive
```

Each output directory holds `image.bin` and `meta.json`. `--lang-tokens` is the number of language tokens of the task prompts, the same as `--lang-tokens` of `scripts/prepare_smolvla.py` (12 by default).

### 6. Bitstream and board program

```
source <Vivado 2025.2>/settings64.sh
cmake --preset gcc-release
cmake --build --preset gcc-release --target fpga-bitstream
cmake --preset aarch64-release
cmake --build --preset aarch64-release
```

`fpga-bitstream` writes `npu.bit.bin`, `npu.dtbo` and `shell.json` to `build/gcc-release/fpga/`. If the KV260 SOM board files are not part of the Vivado installation, configure with `-DHOTARUBI_BOARD_REPO=<board repository>`. The board program is `build/aarch64-release/tools/runtime/hotarubi_run`, linked statically.

### 7. Inputs and references

```
build/.venv/bin/python scripts/prepare_smolvla.py
build/.venv/bin/python scripts/prepare_sparsedrive.py
```

They write the input frames and the outputs of the FP32 models:

- `build/board/smolvla/`: `frames/`, `reference/` and `language.safetensors` (the language embedding rows of the tasks), for 16 frames from the odd-numbered episodes.
- `build/board/sparsedrive/`: `frames/` and `reference/`, for the 81 frames of the nuScenes mini_val scenes scene-0103 and scene-0916.

### 8. Run on the board

```
scp build/gcc-release/fpga/npu.bit.bin build/gcc-release/fpga/npu.dtbo build/gcc-release/fpga/shell.json \
    kv260:/lib/firmware/xilinx/npu/
ssh kv260 'sudo xmutil unloadapp; sudo xmutil loadapp npu'
ssh kv260 'mkdir -p hotarubi/bin hotarubi/compiled hotarubi/inputs/smolvla hotarubi/inputs/sparsedrive'
rsync -a build/aarch64-release/tools/runtime/hotarubi_run kv260:hotarubi/bin/
rsync -a build/compiled/smolvla build/compiled/sparsedrive kv260:hotarubi/compiled/
rsync -a build/board/smolvla/frames build/board/smolvla/language.safetensors kv260:hotarubi/inputs/smolvla/
rsync -a build/board/sparsedrive/frames kv260:hotarubi/inputs/sparsedrive/
ssh kv260 'cd hotarubi && bin/hotarubi_run smolvla --program compiled/smolvla \
    --language inputs/smolvla/language.safetensors --input inputs/smolvla/frames --output outputs/smolvla'
ssh kv260 'cd hotarubi && bin/hotarubi_run sparsedrive --program compiled/sparsedrive \
    --input inputs/sparsedrive/frames --output outputs/sparsedrive'
rsync -a kv260:hotarubi/outputs/smolvla/ build/board/smolvla/outputs/
rsync -a kv260:hotarubi/outputs/sparsedrive/ build/board/sparsedrive/outputs/
```

`hotarubi_run` prints the NPU cycles and the time of each frame.

### 9. Comparison

```
build/.venv/bin/python scripts/compare_board.py smolvla \
    --outputs build/board/smolvla/outputs --references build/board/smolvla/reference
build/.venv/bin/python scripts/compare_board.py sparsedrive \
    --outputs build/board/sparsedrive/outputs --references build/board/sparsedrive/reference
```

- SmolVLA: cosine similarity and largest absolute difference of the actions of each frame, and the mean and minimum cosine similarity.
- SparseDrive-S: mean L2 distance of the planned trajectories to the FP32 trajectories, for all modes and for the selected trajectory; how often the best mode of the driving command agrees; and the detections with a score of at least 0.5, matched to the FP32 detections within 1 m with the same class.

## License

MIT. See [LICENSE](LICENSE).
