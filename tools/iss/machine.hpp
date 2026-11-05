#ifndef HOTARUBI_TOOLS_ISS_MACHINE_HPP_
#define HOTARUBI_TOOLS_ISS_MACHINE_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "ddr.hpp"
#include "dma_engine.hpp"
#include "gemm_array.hpp"
#include "memories.hpp"
#include "vec_engine.hpp"
#include "word.hpp"

namespace hotarubi::iss {

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

class Machine {
 public:
  Machine(std::span<std::byte> ddr, uint64_t ddr_base) : ddr_(ddr, ddr_base) {}

  [[nodiscard]] uint32_t ReadRegister(uint32_t offset) const;
  void WriteRegister(uint32_t offset, uint32_t value);
  Stop Run();
  [[nodiscard]] bool Interrupt() const;

  [[nodiscard]] Memories& OnChip() { return memories_; }
  [[nodiscard]] const VecEngine& Vec() const { return vec_; }
  [[nodiscard]] const DmaEngine& Dma() const { return dma_; }

 private:
  void Start();
  void Issue(uint32_t opcode, const Word128& instruction);
  [[nodiscard]] Word128 Fetch(uint32_t index) const;
  [[nodiscard]] bool WaitingForHost() const;
  [[nodiscard]] uint32_t Status() const;

  Ddr ddr_;
  Memories memories_;
  DmaEngine dma_;
  VecEngine vec_;
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
  uint32_t host_go_ = 0;
  bool trace_enable_ = false;
  bool running_ = false;
  bool halted_ = false;
  bool error_ = false;
};

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_MACHINE_HPP_
