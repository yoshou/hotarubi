#ifndef HOTARUBI_TOOLS_COMPILER_PROGRAM_HPP_
#define HOTARUBI_TOOLS_COMPILER_PROGRAM_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "isa.hpp"
#include "schedule.hpp"

namespace hotarubi::compiler {

enum class Engine : uint8_t {
  kDma = 0,
  kGemm = 1,
  kVec = 2,
  kCtrl = 3,
  kWait = 4,
};

struct Range {
  Mem mem = Mem::kFbuf;
  uint32_t lo = 0;
  uint32_t hi = 0;
};

struct Place {
  uint32_t tile = kNoTile;
  uint32_t offset = 0;

  [[nodiscard]] static constexpr Place Fixed(uint32_t address) {
    return {.tile = kNoTile, .offset = address};
  }
  [[nodiscard]] constexpr Place operator+(uint32_t words) const {
    return {.tile = tile, .offset = offset + words};
  }
};

class Tile {
 public:
  Tile() = default;

  [[nodiscard]] Mem Memory() const { return mem_; }
  [[nodiscard]] uint32_t Words() const { return words_; }
  [[nodiscard]] Place At(uint32_t offset = 0) const {
    return {.tile = id_, .offset = offset};
  }

 private:
  friend class Program;

  Tile(uint32_t id, Mem mem, uint32_t words)
      : id_(id), mem_(mem), words_(words) {}

  uint32_t id_ = kNoTile;
  Mem mem_ = Mem::kFbuf;
  uint32_t words_ = 0;
};

struct Access {
  Mem mem = Mem::kFbuf;
  Place at{};
  uint32_t words = 0;
};

[[nodiscard]] Access Whole(const Tile& tile);
[[nodiscard]] Access Part(const Tile& tile, uint32_t offset, uint32_t words);
[[nodiscard]] Access Fixed(Mem mem, uint32_t address, uint32_t words);

struct Transfer {
  uint64_t ddr_offset = 0;
  Mem mem = Mem::kFbuf;
  Place onchip{};
  uint32_t rows = 1;
  uint32_t row_words = 1;
  uint32_t ddr_stride = 0;
  DmaKind kind = DmaKind::kRaw;
  uint32_t tok = 0;
};

struct GaccArgs {
  uint64_t list_offset = 0;
  Place dst{};
  uint32_t slots = 0;
  uint32_t blocks = 1;
  uint32_t coefficient_stride = 0;
  bool acc = false;
  bool flush = true;
};

struct GemmArgs {
  Place a_base{};
  uint32_t a_stride = 0;
  uint32_t kb = 0;
  uint32_t m = 0;
  Place w_base{};
  uint32_t w_stride = 0;
  uint32_t n8 = 0;
  Place c_base{};
  WeightMode wmode = WeightMode::kFp4;
  bool acc = false;
};

struct VecArgs {
  VecOp op = VecOp::kCopy;
  Place src{};
  Place src2{};
  Place dst{};
  uint32_t rows_q = 1;
  uint32_t n8 = 1;
  Place p_base{};
  uint32_t aux = 0;
  uint32_t imm = 0;
  bool q = false;
  std::optional<Place> scratch = std::nullopt;
};

struct Packed {
  Place place{};
  uint32_t shift = 0;
};

struct RunArgs {
  uint32_t pc = 0;
  Place src{};
  Place src2{};
  Place dst{};
  uint32_t rows_q = 0;
  uint32_t n8 = 0;
  Place p_base{};
  uint32_t imm = 0;
  std::vector<Packed> imm_places;
  std::vector<Access> reads;
  std::vector<Access> writes;
};

class Program {
 public:
  class Scope {
   public:
    Scope(Program& program, std::string_view tag);
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    Scope(Scope&&) = delete;
    Scope& operator=(Scope&&) = delete;
    ~Scope();

   private:
    Program& program_;
  };

  explicit Program(uint64_t base);

  [[nodiscard]] Tile Alloc(Mem mem, uint32_t words);
  size_t Load(const Transfer& transfer, std::string_view note = {});
  size_t Store(const Transfer& transfer, std::string_view note = {});
  size_t Gacc(const GaccArgs& args, std::string_view note = {});
  size_t Gemm(const GemmArgs& args, std::string_view note = {});
  size_t Vec(const VecArgs& args, std::string_view note = {});
  size_t Run(const RunArgs& args, std::string_view note = {});
  size_t Signal(uint32_t id, uint32_t value, std::string_view note = {});
  size_t WaitHost(uint32_t value, std::string_view note = {});
  size_t Halt(std::string_view note = "halt");
  void SetDdrOffset(size_t index, uint64_t ddr_offset);
  void Finalize(const ScheduleOptions& options = {});

  [[nodiscard]] uint64_t Base() const { return base_; }
  [[nodiscard]] size_t Size() const { return nodes_.size(); }
  [[nodiscard]] size_t WaitCount() const;
  [[nodiscard]] size_t EngineCount(Engine engine) const;
  [[nodiscard]] const std::vector<Instruction>& Instructions() const;
  [[nodiscard]] std::vector<std::byte> Bytes() const;
  [[nodiscard]] std::string Listing() const;
  [[nodiscard]] std::string ListingJson() const;
  [[nodiscard]] Engine EngineOf(size_t index) const;
  [[nodiscard]] std::string_view Tag(size_t index) const;
  [[nodiscard]] std::string_view Note(size_t index) const;
  [[nodiscard]] const std::vector<Range>& Reads(size_t index) const;
  [[nodiscard]] const std::vector<Range>& Writes(size_t index) const;
  [[nodiscard]] uint32_t Address(const Tile& tile) const;

 private:
  using Resolver = std::function<uint32_t(Place)>;
  using Maker = std::function<Instruction(const Resolver&, uint32_t)>;

  struct Node {
    Engine engine = Engine::kDma;
    Maker make = nullptr;
    std::vector<Access> reads;
    std::vector<Access> writes;
    uint32_t tag = 0;
    std::string note;
    bool loaded = false;
    std::optional<DmaFields> dma = std::nullopt;
    Place dma_onchip{};
    bool dma_store = false;
    uint64_t ddr_offset = 0;
  };

  struct Done {
    std::vector<Instruction> instructions;
    std::vector<Engine> engines;
    std::vector<uint32_t> tags;
    std::vector<std::string> notes;
    std::vector<std::vector<Range>> reads;
    std::vector<std::vector<Range>> writes;
    std::vector<uint32_t> base;
    size_t waits = 0;
  };

  size_t Emit(Node node);
  size_t Dma(const Transfer& transfer, DmaFields fields, bool store,
             std::vector<Access> reads, std::vector<Access> writes,
             std::string_view note);
  void Check(const Access& access) const;
  [[nodiscard]] Instruction Encode(const Node& node, const Resolver& resolve,
                                   uint32_t wait) const;
  [[nodiscard]] uint32_t DmaAddress(uint64_t offset) const;
  [[nodiscard]] const Done& Finished() const;
  [[nodiscard]] std::vector<Step> Steps(
      const iss::TimingParameters& timing) const;

  void PushScope(std::string_view tag);
  void PopScope();

  uint64_t base_;
  std::vector<Node> nodes_;
  std::vector<TileSize> tiles_;
  std::vector<std::string> tag_names_;
  std::map<std::string, uint32_t, std::less<>> tag_lookup_;
  std::vector<size_t> scope_lengths_;
  std::string scope_;
  std::optional<Done> done_;
};

}  // namespace hotarubi::compiler

#endif  // HOTARUBI_TOOLS_COMPILER_PROGRAM_HPP_
