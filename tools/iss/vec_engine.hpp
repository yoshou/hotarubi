#ifndef HOTARUBI_TOOLS_ISS_VEC_ENGINE_HPP_
#define HOTARUBI_TOOLS_ISS_VEC_ENGINE_HPP_

#include <array>
#include <bitset>
#include <cstdint>

#include "lane.hpp"
#include "memories.hpp"
#include "word.hpp"

namespace hotarubi::iss {

inline constexpr uint32_t kVecRun = 16;
inline constexpr uint32_t kContexts = 2;
inline constexpr uint64_t kOverlapLatencyWords = 10;
inline constexpr uint64_t kReductionLatencyWords = 16;
inline constexpr uint32_t kRowScalarKinds = 8;
inline constexpr uint32_t kScalarArithmeticCycles = 11;
inline constexpr uint32_t kScalarImmediateCycles = 2;
inline constexpr uint32_t kScalarStartCycles = 2;

struct ScalarProgram {
  uint32_t arithmetic = 0;
  uint32_t immediate = 0;
};

inline constexpr std::array<ScalarProgram, kRowScalarKinds> kScalarPrograms = {
    {{.arithmetic = 0, .immediate = 0},
     {.arithmetic = 8, .immediate = 1},
     {.arithmetic = 29, .immediate = 2},
     {.arithmetic = 0, .immediate = 1},
     {.arithmetic = 19, .immediate = 2},
     {.arithmetic = 7, .immediate = 1},
     {.arithmetic = 0, .immediate = 1},
     {.arithmetic = 19, .immediate = 2}}};
inline constexpr ScalarProgram kCountReciprocalProgram = {.arithmetic = 12,
                                                          .immediate = 2};

constexpr uint32_t ScalarProgramCycles(const ScalarProgram& program) {
  return (program.arithmetic * kScalarArithmeticCycles) +
         (program.immediate * kScalarImmediateCycles) + kScalarStartCycles;
}

struct VecActivity {
  uint32_t passes = 0;
  uint64_t words = 0;
  uint32_t drains = 0;
  uint32_t drains_quantized = 0;
  uint32_t overlaps = 0;
  uint64_t overlap_stall_words = 0;
  uint64_t context_stall_words = 0;
  std::array<uint32_t, kRowScalarKinds> scalars{};
  uint32_t rn = 0;
  uint32_t salu = 0;
  uint32_t smem = 0;
  uint32_t branches = 0;
  uint32_t transposes = 0;
  uint32_t transpose_tiles = 0;

  friend constexpr bool operator==(const VecActivity&,
                                   const VecActivity&) = default;
};

struct VecCommand {
  uint32_t op = 0;
  bool quantize = false;
  uint32_t src = 0;
  uint32_t src2 = 0;
  uint32_t dst = 0;
  uint32_t rows_q = 0;
  uint32_t n8 = 0;
  uint32_t p_base = 0;
  uint32_t aux = 0;
  uint32_t imm = 0;
};

class VecEngine {
 public:
  static constexpr uint32_t kRegisters = 32;

  void Execute(const VecCommand& command, Memories& memories);

  [[nodiscard]] uint32_t Register(uint32_t index) const;
  [[nodiscard]] const RowState& Rows(uint32_t ctx) const {
    return rows_.at(ctx);
  }
  [[nodiscard]] uint32_t Rn() const { return rn_; }
  [[nodiscard]] const VecActivity& Activity() const { return activity_; }

 private:
  struct PassState;

  void Initialize(const VecCommand& command);
  void RunPass(const Word128& instruction, const Word128& next,
               Memories& memories);
  void RunWord(PassState& state, uint32_t w, Memories& memories);
  [[nodiscard]] const FbufWord& Operand(const PassState& state,
                                        uint32_t address, Memories& memories);
  void Store(PassState& state, uint32_t word, const FbufWord& out,
             Memories& memories);
  void RunSalu(const Word128& instruction);
  void RunTranspose(const Word128& instruction, Memories& memories);
  void RunSmem(const Word128& instruction, Memories& memories);
  [[nodiscard]] bool Taken(const Word128& instruction) const;
  void Drain();
  void Write(uint32_t index, uint32_t value);
  void AwaitContext(uint32_t ctx);

  std::array<uint32_t, kRegisters> registers_{};
  VecActivity activity_;
  std::array<RowState, kContexts> rows_{};
  uint32_t rn_ = 0;
  bool quantize_ = false;
  std::bitset<kFbufDepth> in_flight_;
  std::bitset<kFbufDepth> pending_;
  std::array<uint64_t, kFbufDepth> pending_slot_{};
  uint64_t slot_ = 0;
  std::array<uint64_t, kContexts> ready_slot_{};
  uint64_t scalar_free_slot_ = 0;
  bool undrained_ = false;
  bool undrained_pbuf_at_mul_ = false;
};

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_VEC_ENGINE_HPP_
