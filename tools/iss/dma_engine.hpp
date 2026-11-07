#ifndef HOTARUBI_TOOLS_ISS_DMA_ENGINE_HPP_
#define HOTARUBI_TOOLS_ISS_DMA_ENGINE_HPP_

#include <array>
#include <cstdint>

#include "ddr.hpp"
#include "memories.hpp"

namespace hotarubi::iss {

inline constexpr uint32_t kGatherChannels = 256;

struct DmaActivity {
  uint64_t read_beats = 0;
  uint64_t write_beats = 0;
  uint64_t read_bursts = 0;
  uint64_t write_bursts = 0;
  uint64_t null_entries = 0;
  uint64_t gather_entries = 0;
  uint64_t gather_rows = 0;

  friend constexpr bool operator==(const DmaActivity&,
                                   const DmaActivity&) = default;
};

struct DmaCommand {
  bool store = false;
  uint32_t ddr_addr = 0;
  uint32_t mem = 0;
  uint32_t onchip = 0;
  uint32_t rows = 0;
  uint32_t row_words = 0;
  uint32_t ddr_stride = 0;
  uint32_t mode = 0;
  uint32_t flags = 0;
  uint32_t tok = 0;
};

uint32_t Int8ToFp32(uint8_t value);
uint32_t ScaleCoefficient(uint32_t coefficient, uint32_t scale);
uint8_t AlignElement(uint8_t value, uint32_t shift);

class DmaEngine {
 public:
  void Execute(const DmaCommand& command, Ddr& ddr, Memories& memories);

  [[nodiscard]] const std::array<std::array<uint32_t, kGatherChannels>,
                                 kTokens>&
  Accumulators() const {
    return accumulators_;
  }
  [[nodiscard]] const DmaActivity& Activity() const { return activity_; }

 private:
  void Gather(const DmaCommand& command, const Ddr& ddr, Memories& memories);

  std::array<std::array<uint32_t, kGatherChannels>, kTokens> accumulators_{};
  DmaActivity activity_;
};

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_DMA_ENGINE_HPP_
