#ifndef HOTARUBI_TOOLS_COMPILER_PROGRAM_HPP_
#define HOTARUBI_TOOLS_COMPILER_PROGRAM_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "isa.hpp"

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

struct Transfer {
  uint64_t ddr_offset = 0;
  Mem mem = Mem::kFbuf;
  uint32_t onchip = 0;
  uint32_t rows = 1;
  uint32_t row_words = 1;
  uint32_t ddr_stride = 0;
  DmaMode mode = DmaMode::kRaw;
  bool index = false;
  bool row_format = false;
  uint32_t tok = 0;
};

struct GaccArgs {
  uint64_t list_offset = 0;
  uint32_t dst = 0;
  uint32_t slots = 0;
  uint32_t blocks = 1;
  uint32_t quad = 0;
  bool acc = false;
  bool flush = true;
};

struct GemmArgs {
  uint32_t a_base = 0;
  uint32_t a_stride = 0;
  uint32_t kb = 0;
  uint32_t m = 0;
  uint32_t w_base = 0;
  uint32_t w_stride = 0;
  uint32_t n8 = 0;
  uint32_t c_base = 0;
  WeightMode wmode = WeightMode::kFp4;
  bool acc = false;
};

struct VecArgs {
  VecOp op = VecOp::kCopy;
  uint32_t src = 0;
  uint32_t src2 = 0;
  uint32_t dst = 0;
  uint32_t rows_q = 1;
  uint32_t n8 = 1;
  uint32_t p_base = 0;
  uint32_t aux = 0;
  uint32_t imm = 0;
  bool q = false;
  std::optional<uint32_t> scratch = std::nullopt;
};

struct RunArgs {
  uint32_t pc = 0;
  uint32_t src = 0;
  uint32_t src2 = 0;
  uint32_t dst = 0;
  uint32_t rows_q = 0;
  uint32_t n8 = 0;
  uint32_t p_base = 0;
  uint32_t imm = 0;
  std::optional<std::vector<Range>> reads = std::nullopt;
  std::optional<std::vector<Range>> writes = std::nullopt;
};

class Program {
 public:
  explicit Program(uint64_t base);

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

  [[nodiscard]] uint64_t Base() const { return base_; }
  [[nodiscard]] size_t Size() const { return instructions_.size(); }
  [[nodiscard]] size_t WaitCount() const { return wait_count_; }
  [[nodiscard]] size_t EngineCount(Engine engine) const;
  [[nodiscard]] const std::vector<Instruction>& Instructions() const {
    return instructions_;
  }
  [[nodiscard]] std::vector<std::byte> Bytes() const;
  [[nodiscard]] std::string Listing() const;

 private:
  static constexpr int kEngines = 3;

  struct Tracker {
    std::vector<int32_t> write_index;
    std::vector<int8_t> write_engine;
    std::array<std::vector<int32_t>, kEngines> read_index;
  };

  static int32_t LatestWrite(const Tracker& tracker, const Range& range,
                             int engine);
  static int32_t LatestRead(const Tracker& tracker, const Range& range,
                            int engine);
  [[nodiscard]] uint32_t Dependencies(int engine,
                                      const std::vector<Range>& reads,
                                      const std::vector<Range>& writes) const;
  size_t Emit(Engine engine, const std::function<Instruction(uint32_t)>& make,
              const std::vector<Range>& reads, const std::vector<Range>& writes,
              std::string_view note);
  [[nodiscard]] uint32_t DmaAddress(uint64_t offset) const;

  uint64_t base_;
  std::vector<Instruction> instructions_;
  std::vector<std::string> notes_;
  std::vector<Engine> engines_;
  std::array<int32_t, kEngines> known_{};
  std::array<Tracker, kMemCount> trackers_;
  size_t wait_count_ = 0;
};

}  // namespace hotarubi::compiler

#endif  // HOTARUBI_TOOLS_COMPILER_PROGRAM_HPP_
