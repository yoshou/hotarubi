#ifndef HOTARUBI_TOOLS_COMPILER_KERNELS_HPP_
#define HOTARUBI_TOOLS_COMPILER_KERNELS_HPP_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "isa.hpp"
#include "program.hpp"

namespace hotarubi::compiler {

struct ValidKeys {
  uint32_t quad = 0;
  uint32_t keys = 0;
};

struct Attention {
  uint64_t q = 0;
  std::optional<Tile> queries = std::nullopt;
  uint64_t k = 0;
  uint64_t v = 0;
  uint64_t o = 0;
  uint32_t heads = 0;
  uint32_t kv_heads = 0;
  uint32_t rows = 0;
  uint32_t chunk_quads = 8;
  uint32_t embed = 0;
  uint32_t kv_blocks = 0;
  uint32_t hd = 0;
  uint32_t keys = 0;
  std::vector<ValidKeys> valid;
  bool chunk_local = false;
  bool scale = false;
  std::optional<uint64_t> mask = std::nullopt;
  uint32_t mask_stride = 0;
};

void EmitAttention(Program& program, const Attention& attention,
                   std::string_view note);

struct Quantize {
  uint64_t x = 0;
  uint32_t quads = 0;
  uint32_t words = 0;
  uint32_t tile_quads = 8;
  bool tokens = false;
  VecOp op = VecOp::kCopy;
  Place p_base{};
  uint32_t aux = 0;
  uint32_t imm = 0;
  Place dst{};
};

using QuantizeHook =
    std::function<void(const Tile& rows, uint32_t quad, uint32_t quads)>;

void EmitQuantize(Program& program, const Quantize& quantize,
                  std::string_view note, const QuantizeHook& before = {});

struct LinearPlane {
  WeightMode mode = WeightMode::kFp4;
  uint64_t w = 0;
};

struct LinearPiece {
  Place a{};
  uint32_t a_stride = 0;
  uint32_t kb = 0;
  uint32_t kb_offset = 0;
};

struct Linear {
  std::vector<LinearPlane> planes;
  uint32_t kb_total = 0;
  std::vector<LinearPiece> pieces;
  uint32_t quads = 0;
  uint32_t chunk_quads = 0;
  uint32_t groups = 0;
  uint32_t tile_groups = 0;
};

struct LinearTile {
  Tile c;
  uint32_t tile = 0;
  uint32_t chunk = 0;
  uint32_t first = 0;
  uint32_t groups = 0;
  uint32_t quad = 0;
  uint32_t quads = 0;
};

class LinearEmitter {
 public:
  LinearEmitter(Program& program, Linear linear, std::string_view note);

  [[nodiscard]] uint32_t Tiles() const;
  [[nodiscard]] uint32_t Chunks() const;
  LinearTile Emit(uint32_t tile, uint32_t chunk,
                  const std::optional<Tile>& onto = std::nullopt);

 private:
  Program& program_;
  Linear linear_;
  std::string note_;
  std::optional<uint32_t> tile_;
  std::vector<std::optional<Tile>> weights_;
};

void EmitLinear(Program& program, const Linear& linear, std::string_view note,
                const std::function<void(const LinearTile&)>& post);

}  // namespace hotarubi::compiler

#endif  // HOTARUBI_TOOLS_COMPILER_KERNELS_HPP_
