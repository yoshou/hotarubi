#ifndef HOTARUBI_TOOLS_ISS_TIMING_HPP_
#define HOTARUBI_TOOLS_ISS_TIMING_HPP_

#include <array>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

#include "dma_engine.hpp"
#include "gemm_array.hpp"
#include "trace_record.hpp"
#include "vec_engine.hpp"

namespace hotarubi::iss {

inline constexpr uint32_t kBeatsPerKilobeat = 1024;

struct TimingParameters {
  uint32_t start_cycles = 44;
  uint32_t gemm_fixed_cycles = 51;
  uint32_t dma_fixed_cycles = 26;
  uint32_t dma_read_cycles_per_kilobeat = 833;
  uint32_t dma_read_burst_cycles = 3;
  uint32_t dma_write_cycles_per_kilobeat = 939;
  uint32_t dma_write_burst_cycles = 1;
  uint32_t dma_gather_entry_cycles = 0;
  uint32_t dma_gather_row_cycles = 18;
  uint32_t dma_gather_fetch_cycles = 23;
  uint32_t vec_fixed_cycles = 22;
  uint32_t vec_pass_cycles = 2;
  uint32_t vec_word_cycles = 4;
  uint32_t vec_issue_tail_cycles = 2;
  uint32_t vec_drain_cycles = 41;
  uint32_t vec_drain_quantized_cycles = 22;
  uint32_t vec_salu_cycles = 5;
  uint32_t vec_smem_cycles = 0;
  uint32_t vec_branch_cycles = 2;
  uint32_t vec_transpose_cycles = 26;
  uint32_t vec_transpose_tile_cycles = 10;
};

std::expected<TimingParameters, std::string> ParseTiming(
    std::string_view assignments, TimingParameters base = {});

struct Cost {
  uint64_t cycles = 0;
  Statistics statistics;
};

class CostModel {
 public:
  explicit CostModel(const TimingParameters& parameters = {})
      : parameters_(parameters) {}

  [[nodiscard]] const TimingParameters& Parameters() const {
    return parameters_;
  }
  [[nodiscard]] Cost Dma(const DmaActivity& activity) const;
  [[nodiscard]] Cost Gemm(const GemmCommand& command) const;
  [[nodiscard]] Cost Vec(const VecActivity& activity) const;
  [[nodiscard]] uint64_t ScalarCycles(const VecActivity& activity) const;
  [[nodiscard]] uint64_t IssueCycles(const VecActivity& activity) const;
  [[nodiscard]] uint64_t DrainCycles(const VecActivity& activity) const;

 private:
  TimingParameters parameters_;
};

struct DdrExtent {
  uint64_t lo = 0;
  uint64_t hi = 0;
  bool whole = false;

  friend constexpr bool operator==(const DdrExtent&,
                                   const DdrExtent&) = default;
};

uint64_t GemmArrayCycles(const GemmCommand& command);
DmaActivity EstimateDma(const DmaCommand& command);
DdrExtent EstimateDdrExtent(const DmaCommand& command);
VecActivity FixedOperationActivity(const VecCommand& command);

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_TIMING_HPP_
