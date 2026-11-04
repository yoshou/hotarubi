#ifndef HOTARUBI_TB_NPU_BENCH_HPP_
#define HOTARUBI_TB_NPU_BENCH_HPP_

#include <Vnpu_top.h>

#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace hotarubi::tb {

using Beat = std::array<uint32_t, 4>;

class Ddr {
 public:
  static constexpr uint32_t kBase = 0x10000000U;
  static constexpr uint32_t kSize = 0x01000000U;

  Ddr();

  [[nodiscard]] static bool Contains(uint64_t address, uint64_t bytes);
  [[nodiscard]] uint8_t Read8(uint32_t address) const;
  void Write8(uint32_t address, uint8_t value);
  [[nodiscard]] uint32_t Read32(uint32_t address) const;
  void Write32(uint32_t address, uint32_t value);
  [[nodiscard]] Beat ReadBeat(uint32_t address) const;
  void WriteBeat(uint32_t address, const Beat& beat);

 private:
  std::vector<uint8_t> bytes_;
};

struct AxiRequest {
  uint32_t address = 0;
  uint32_t beats = 0;
  uint32_t done = 0;
  uint64_t ready_cycle = 0;
};

struct AxiMasterOutputs {
  bool ar_valid = false;
  uint32_t ar_addr = 0;
  uint32_t ar_len = 0;
  uint32_t ar_size = 0;
  uint32_t ar_burst = 0;
  bool r_ready = false;
  bool aw_valid = false;
  uint32_t aw_addr = 0;
  uint32_t aw_len = 0;
  uint32_t aw_size = 0;
  uint32_t aw_burst = 0;
  bool w_valid = false;
  Beat w_data{};
  uint32_t w_strb = 0;
  bool w_last = false;
  bool b_ready = false;
};

struct AxiSlaveOutputs {
  bool ar_ready = false;
  bool r_valid = false;
  Beat r_data{};
  bool r_last = false;
  bool aw_ready = false;
  bool w_ready = false;
  bool b_valid = false;
};

class AxiSlave {
 public:
  AxiSlave(std::string name, uint32_t max_beats, std::mt19937& rng);

  AxiSlaveOutputs Drive(const Ddr& ddr, uint64_t cycle);
  void Sample(Ddr& ddr, uint64_t cycle, const AxiMasterOutputs& master);
  [[nodiscard]] const std::vector<std::string>& Errors() const {
    return errors_;
  }

 private:
  bool Chance(uint32_t percent);
  void CheckAddress(const char* channel, uint32_t address, uint32_t len,
                    uint32_t size, uint32_t burst);
  void Error(const std::string& message);
  void CheckStability(const AxiMasterOutputs& master);
  void AcceptWriteBeat(Ddr& ddr, uint64_t cycle,
                       const AxiMasterOutputs& master);
  void AcceptAddresses(uint64_t cycle, const AxiMasterOutputs& master);

  std::string name_;
  uint32_t max_beats_;
  std::mt19937& rng_;
  std::deque<AxiRequest> reads_;
  std::deque<AxiRequest> writes_;
  std::deque<uint64_t> responses_;
  AxiSlaveOutputs out_;
  AxiMasterOutputs last_;
  bool last_ar_accepted_ = false;
  bool last_aw_accepted_ = false;
  bool last_w_accepted_ = false;
  std::vector<std::string> errors_;
};

struct Instruction {
  std::array<uint32_t, 4> words{};

  Instruction& Set(uint32_t lsb, uint32_t width, uint64_t value);
};

class NpuBench {
 public:
  static constexpr uint32_t kProgramBase = Ddr::kBase;

  explicit NpuBench(uint32_t seed = 1);
  ~NpuBench();
  NpuBench(const NpuBench&) = delete;
  NpuBench& operator=(const NpuBench&) = delete;
  NpuBench(NpuBench&&) = delete;
  NpuBench& operator=(NpuBench&&) = delete;

  Ddr& Memory() { return ddr_; }
  [[nodiscard]] bool Irq() const;
  [[nodiscard]] std::vector<std::string> Errors() const;

  void Tick();
  void WriteReg(uint32_t offset, uint32_t value);
  uint32_t ReadReg(uint32_t offset);
  void LoadProgram(std::span<const Instruction> program, uint32_t address);
  void Start(uint32_t address, uint32_t count);
  bool WaitStatus(const std::function<bool(uint32_t)>& done,
                  uint64_t max_cycles);
  bool Run(std::span<const Instruction> program, uint64_t max_cycles = 2000000);

 private:
  [[nodiscard]] AxiMasterOutputs Master0() const;
  [[nodiscard]] AxiMasterOutputs Master1() const;
  void ApplySlave0(const AxiSlaveOutputs& out);
  void ApplySlave1(const AxiSlaveOutputs& out);
  void Settle();
  void Edge();

  std::unique_ptr<Vnpu_top> top_;
  Ddr ddr_;
  std::mt19937 rng_;
  AxiSlave slave0_;
  AxiSlave slave1_;
  uint64_t cycle_ = 0;
};

}  // namespace hotarubi::tb

#endif  // HOTARUBI_TB_NPU_BENCH_HPP_
