#ifndef HOTARUBI_TOOLS_ISS_MACHINE_HPP_
#define HOTARUBI_TOOLS_ISS_MACHINE_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "ddr.hpp"
#include "dma_engine.hpp"
#include "gemm_array.hpp"
#include "memories.hpp"
#include "timeline.hpp"
#include "timing.hpp"
#include "trace_record.hpp"
#include "vec_engine.hpp"
#include "word.hpp"

namespace hotarubi::iss {

struct Hazard {
  uint32_t earlier = 0;
  uint32_t later = 0;
  Conflict conflict;
};

std::string DescribeHazards(const std::vector<Hazard>& hazards, size_t limit);

enum class Stop : uint8_t {
  kIdle,
  kSignal,
  kHalt,
  kIllegal,
  kWaitHost,
  kStarved,
};

DmaCommand DecodeDma(const Word128& instruction);
GemmCommand DecodeGemm(const Word128& instruction);
VecCommand DecodeVec(const Word128& instruction);
uint64_t EstimateCycles(const Word128& instruction, const CostModel& model);

class Machine {
 public:
  Machine(std::span<std::byte> ddr, uint64_t ddr_base,
          const TimingParameters& timing = {})
      : ddr_(ddr, ddr_base), model_(timing), timeline_(timing) {}

  [[nodiscard]] uint32_t ReadRegister(uint32_t offset) const;
  void WriteRegister(uint32_t offset, uint32_t value);
  Stop Run();
  [[nodiscard]] bool Interrupt() const;

  [[nodiscard]] Memories& OnChip() { return memories_; }
  [[nodiscard]] const VecEngine& Vec() const { return vec_; }
  [[nodiscard]] const DmaEngine& Dma() const { return dma_; }
  [[nodiscard]] const Timeline& Time() const { return timeline_; }
  [[nodiscard]] const std::vector<Hazard>& Hazards() const { return hazards_; }
  void Observe(std::function<void(uint32_t, const AccessLog&)> observer) {
    observer_ = std::move(observer);
  }

 private:
  void Start();
  void Issue(uint32_t opcode, const Word128& instruction);
  [[nodiscard]] Cost Execute(uint32_t opcode, const Word128& instruction,
                             uint64_t& earliest_start);
  void Record(const Word128& instruction, const Span& span,
              const Statistics& statistics);
  void CheckHazards(Source source, uint32_t wait, const AccessLog& log);
  [[nodiscard]] Word128 Fetch(uint32_t index) const;
  [[nodiscard]] bool WaitingForHost() const;
  [[nodiscard]] uint32_t Status() const;

  Ddr ddr_;
  Memories memories_;
  DmaEngine dma_;
  VecEngine vec_;
  CostModel model_;
  Timeline timeline_;
  struct IssuedAccess {
    uint32_t index = 0;
    AccessLog log;
  };
  std::array<std::vector<IssuedAccess>, Timeline::kEngines> history_;
  std::array<uint32_t, Timeline::kEngines> last_{};
  std::array<uint32_t, Timeline::kEngines> known_{};
  std::vector<Hazard> hazards_;
  std::function<void(uint32_t, const AccessLog&)> observer_;
  uint32_t instr_base_ = 0;
  uint32_t instr_count_ = 0;
  uint32_t stream_base_ = 0;
  uint32_t stream_count_ = 0;
  uint32_t next_ = 0;
  uint32_t irq_status_ = 0;
  uint32_t irq_enable_ = 0;
  uint32_t signal_id_ = 0;
  uint32_t signal_count_ = 0;
  uint32_t signal_value_ = 0;
  uint32_t issued_ = 0;
  uint32_t trace_base_ = 0;
  uint32_t trace_count_ = 0;
  uint32_t host_go_ = 0;
  bool trace_enable_ = false;
  bool trace_statistics_ = false;
  bool running_ = false;
  bool halted_ = false;
  bool error_ = false;
};

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_MACHINE_HPP_
