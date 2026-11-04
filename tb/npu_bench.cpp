#include "npu_bench.hpp"

#include <verilated.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace hotarubi::tb {

namespace {

constexpr uint32_t kRegCtrl = 0x00;
constexpr uint32_t kRegStatus = 0x04;
constexpr uint32_t kRegInstrBase = 0x08;
constexpr uint32_t kRegInstrCount = 0x0C;
constexpr uint32_t kStatusHalted = 0x2;
constexpr uint32_t kStatusError = 0x4;
constexpr int kHandshakeLimit = 10000;

Beat ToBeat(const VlWide<4>& wide) {
  return {wide[0], wide[1], wide[2], wide[3]};
}

void FromBeat(VlWide<4>& wide, const Beat& beat) {
  for (uint32_t i = 0; i < 4; ++i) {
    wide[i] = beat[i];
  }
}

}  // namespace

Ddr::Ddr() : bytes_(kSize, 0) {}

bool Ddr::Contains(uint64_t address, uint64_t bytes) {
  return address >= kBase && address + bytes <= uint64_t{kBase} + kSize;
}

uint8_t Ddr::Read8(uint32_t address) const {
  return bytes_.at(address - kBase);
}

void Ddr::Write8(uint32_t address, uint8_t value) {
  bytes_.at(address - kBase) = value;
}

uint32_t Ddr::Read32(uint32_t address) const {
  uint32_t value = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    value |= static_cast<uint32_t>(Read8(address + i)) << (8 * i);
  }
  return value;
}

void Ddr::Write32(uint32_t address, uint32_t value) {
  for (uint32_t i = 0; i < 4; ++i) {
    Write8(address + i, static_cast<uint8_t>(value >> (8 * i)));
  }
}

Beat Ddr::ReadBeat(uint32_t address) const {
  Beat beat{};
  for (uint32_t i = 0; i < 4; ++i) {
    beat.at(i) = Read32(address + (4 * i));
  }
  return beat;
}

void Ddr::WriteBeat(uint32_t address, const Beat& beat) {
  for (uint32_t i = 0; i < 4; ++i) {
    Write32(address + (4 * i), beat.at(i));
  }
}

AxiSlave::AxiSlave(std::string name, uint32_t max_beats, std::mt19937& rng)
    : name_(std::move(name)), max_beats_(max_beats), rng_(rng) {}

bool AxiSlave::Chance(uint32_t percent) { return rng_() % 100 < percent; }

void AxiSlave::Error(const std::string& message) {
  errors_.push_back(name_ + ": " + message);
}

void AxiSlave::CheckAddress(const char* channel, uint32_t address, uint32_t len,
                            uint32_t size, uint32_t burst) {
  const uint32_t beats = len + 1;
  if (size != 4) {
    Error(std::string(channel) + " size " + std::to_string(size));
  }
  if (burst != 1) {
    Error(std::string(channel) + " burst " + std::to_string(burst));
  }
  if (beats > max_beats_) {
    Error(std::string(channel) + " beats " + std::to_string(beats));
  }
  if ((address & 0xFU) != 0) {
    Error(std::string(channel) + " unaligned " + std::to_string(address));
  }
  if ((address & 0xFFFU) + (16 * beats) > 0x1000U) {
    Error(std::string(channel) + " crosses 4 KB at " + std::to_string(address));
  }
}

AxiSlaveOutputs AxiSlave::Drive(const Ddr& ddr, uint64_t cycle) {
  out_.ar_ready = Chance(70);
  out_.aw_ready = Chance(70);
  out_.w_ready = !writes_.empty() && Chance(75);
  const bool hold = out_.r_valid && !last_.r_ready;
  if (!hold) {
    out_.r_valid =
        !reads_.empty() && reads_.front().ready_cycle <= cycle && Chance(80);
    out_.r_data = Beat{};
    out_.r_last = false;
    if (out_.r_valid) {
      const AxiRequest& request = reads_.front();
      const uint32_t address = request.address + (16 * request.done);
      if (Ddr::Contains(address, 16)) {
        out_.r_data = ddr.ReadBeat(address);
      }
      out_.r_last = request.done + 1 == request.beats;
    }
  }
  out_.b_valid = !responses_.empty() && responses_.front() <= cycle;
  return out_;
}

void AxiSlave::CheckStability(const AxiMasterOutputs& master) {
  if (last_.ar_valid && !last_ar_accepted_ &&
      (!master.ar_valid || master.ar_addr != last_.ar_addr ||
       master.ar_len != last_.ar_len)) {
    Error("AR changed before acceptance");
  }
  if (last_.aw_valid && !last_aw_accepted_ &&
      (!master.aw_valid || master.aw_addr != last_.aw_addr ||
       master.aw_len != last_.aw_len)) {
    Error("AW changed before acceptance");
  }
  if (last_.w_valid && !last_w_accepted_ &&
      (!master.w_valid || master.w_data != last_.w_data ||
       master.w_last != last_.w_last)) {
    Error("W changed before acceptance");
  }
  last_ar_accepted_ = master.ar_valid && out_.ar_ready;
  last_aw_accepted_ = master.aw_valid && out_.aw_ready;
  last_w_accepted_ = master.w_valid && out_.w_ready;
}

void AxiSlave::AcceptWriteBeat(Ddr& ddr, uint64_t cycle,
                               const AxiMasterOutputs& master) {
  AxiRequest& request = writes_.front();
  const uint32_t address = request.address + (16 * request.done);
  if (master.w_last != (request.done + 1 == request.beats)) {
    Error("WLAST mismatch");
  }
  if (!Ddr::Contains(address, 16)) {
    Error("write outside DDR at " + std::to_string(address));
  } else {
    for (uint32_t byte = 0; byte < 16; ++byte) {
      if (((master.w_strb >> byte) & 1U) != 0) {
        ddr.Write8(address + byte,
                   static_cast<uint8_t>(master.w_data.at(byte / 4) >>
                                        (8 * (byte % 4))));
      }
    }
  }
  if (++request.done == request.beats) {
    writes_.pop_front();
    responses_.push_back(cycle + 1 + (rng_() % 16));
  }
}

void AxiSlave::AcceptAddresses(uint64_t cycle, const AxiMasterOutputs& master) {
  if (master.ar_valid && out_.ar_ready) {
    CheckAddress("AR", master.ar_addr, master.ar_len, master.ar_size,
                 master.ar_burst);
    if (!Ddr::Contains(master.ar_addr, 16 * (uint64_t{master.ar_len} + 1))) {
      Error("read outside DDR at " + std::to_string(master.ar_addr));
    }
    reads_.push_back({.address = master.ar_addr,
                      .beats = master.ar_len + 1,
                      .done = 0,
                      .ready_cycle = cycle + 1 + (rng_() % 64)});
  }
  if (master.aw_valid && out_.aw_ready) {
    CheckAddress("AW", master.aw_addr, master.aw_len, master.aw_size,
                 master.aw_burst);
    writes_.push_back({.address = master.aw_addr,
                       .beats = master.aw_len + 1,
                       .done = 0,
                       .ready_cycle = 0});
  }
}

void AxiSlave::Sample(Ddr& ddr, uint64_t cycle,
                      const AxiMasterOutputs& master) {
  CheckStability(master);
  if (out_.r_valid && master.r_ready) {
    AxiRequest& request = reads_.front();
    if (++request.done == request.beats) {
      reads_.pop_front();
    }
  }
  if (master.w_valid && out_.w_ready) {
    AcceptWriteBeat(ddr, cycle, master);
  }
  AcceptAddresses(cycle, master);
  if (out_.b_valid && master.b_ready) {
    responses_.pop_front();
  }
  last_ = master;
}

Instruction& Instruction::Set(uint32_t lsb, uint32_t width, uint64_t value) {
  for (uint32_t i = 0; i < width; ++i) {
    const uint32_t position = lsb + i;
    const uint32_t mask = 1U << (position % 32);
    uint32_t& word = words.at(position / 32);
    word = ((value >> i) & 1U) != 0 ? (word | mask) : (word & ~mask);
  }
  return *this;
}

NpuBench::NpuBench(uint32_t seed)
    : top_(std::make_unique<Vnpu_top>()),
      rng_(seed),
      slave0_("m0", 16, rng_),
      slave1_("m1", 256, rng_) {
  top_->clk_i = 0;
  top_->rst_ni = 0;
  top_->s_axil_awaddr_i = 0;
  top_->s_axil_awvalid_i = 0;
  top_->s_axil_wdata_i = 0;
  top_->s_axil_wstrb_i = 0xF;
  top_->s_axil_wvalid_i = 0;
  top_->s_axil_bready_i = 0;
  top_->s_axil_araddr_i = 0;
  top_->s_axil_arvalid_i = 0;
  top_->s_axil_rready_i = 0;
  for (int i = 0; i < 8; ++i) {
    Tick();
  }
  top_->rst_ni = 1;
  Tick();
}

NpuBench::~NpuBench() { top_->final(); }

bool NpuBench::Irq() const { return top_->irq_o != 0; }

std::vector<std::string> NpuBench::Errors() const {
  std::vector<std::string> errors = slave0_.Errors();
  errors.insert(errors.end(), slave1_.Errors().begin(), slave1_.Errors().end());
  return errors;
}

AxiMasterOutputs NpuBench::Master0() const {
  return {.ar_valid = top_->m0_axi_arvalid_o != 0,
          .ar_addr = top_->m0_axi_araddr_o,
          .ar_len = top_->m0_axi_arlen_o,
          .ar_size = top_->m0_axi_arsize_o,
          .ar_burst = top_->m0_axi_arburst_o,
          .r_ready = top_->m0_axi_rready_o != 0,
          .aw_valid = top_->m0_axi_awvalid_o != 0,
          .aw_addr = top_->m0_axi_awaddr_o,
          .aw_len = top_->m0_axi_awlen_o,
          .aw_size = top_->m0_axi_awsize_o,
          .aw_burst = top_->m0_axi_awburst_o,
          .w_valid = top_->m0_axi_wvalid_o != 0,
          .w_data = ToBeat(top_->m0_axi_wdata_o),
          .w_strb = top_->m0_axi_wstrb_o,
          .w_last = top_->m0_axi_wlast_o != 0,
          .b_ready = top_->m0_axi_bready_o != 0};
}

AxiMasterOutputs NpuBench::Master1() const {
  return {.ar_valid = top_->m1_axi_arvalid_o != 0,
          .ar_addr = top_->m1_axi_araddr_o,
          .ar_len = top_->m1_axi_arlen_o,
          .ar_size = top_->m1_axi_arsize_o,
          .ar_burst = top_->m1_axi_arburst_o,
          .r_ready = top_->m1_axi_rready_o != 0,
          .aw_valid = top_->m1_axi_awvalid_o != 0,
          .aw_addr = top_->m1_axi_awaddr_o,
          .aw_len = top_->m1_axi_awlen_o,
          .aw_size = top_->m1_axi_awsize_o,
          .aw_burst = top_->m1_axi_awburst_o,
          .w_valid = top_->m1_axi_wvalid_o != 0,
          .w_data = ToBeat(top_->m1_axi_wdata_o),
          .w_strb = top_->m1_axi_wstrb_o,
          .w_last = top_->m1_axi_wlast_o != 0,
          .b_ready = top_->m1_axi_bready_o != 0};
}

void NpuBench::ApplySlave0(const AxiSlaveOutputs& out) {
  top_->m0_axi_arready_i = out.ar_ready ? 1 : 0;
  FromBeat(top_->m0_axi_rdata_i, out.r_data);
  top_->m0_axi_rresp_i = 0;
  top_->m0_axi_rlast_i = out.r_last ? 1 : 0;
  top_->m0_axi_rvalid_i = out.r_valid ? 1 : 0;
  top_->m0_axi_awready_i = out.aw_ready ? 1 : 0;
  top_->m0_axi_wready_i = out.w_ready ? 1 : 0;
  top_->m0_axi_bresp_i = 0;
  top_->m0_axi_bvalid_i = out.b_valid ? 1 : 0;
}

void NpuBench::ApplySlave1(const AxiSlaveOutputs& out) {
  top_->m1_axi_arready_i = out.ar_ready ? 1 : 0;
  FromBeat(top_->m1_axi_rdata_i, out.r_data);
  top_->m1_axi_rresp_i = 0;
  top_->m1_axi_rlast_i = out.r_last ? 1 : 0;
  top_->m1_axi_rvalid_i = out.r_valid ? 1 : 0;
  top_->m1_axi_awready_i = out.aw_ready ? 1 : 0;
  top_->m1_axi_wready_i = out.w_ready ? 1 : 0;
  top_->m1_axi_bresp_i = 0;
  top_->m1_axi_bvalid_i = out.b_valid ? 1 : 0;
}

void NpuBench::Settle() {
  ApplySlave0(slave0_.Drive(ddr_, cycle_));
  ApplySlave1(slave1_.Drive(ddr_, cycle_));
  top_->clk_i = 0;
  top_->eval();
}

void NpuBench::Edge() {
  slave0_.Sample(ddr_, cycle_, Master0());
  slave1_.Sample(ddr_, cycle_, Master1());
  top_->clk_i = 1;
  top_->eval();
  ++cycle_;
}

void NpuBench::Tick() {
  Settle();
  Edge();
}

void NpuBench::WriteReg(uint32_t offset, uint32_t value) {
  top_->s_axil_awaddr_i = offset;
  top_->s_axil_awvalid_i = 1;
  top_->s_axil_wdata_i = value;
  top_->s_axil_wvalid_i = 1;
  top_->s_axil_bready_i = 1;
  for (int i = 0; i < kHandshakeLimit; ++i) {
    Settle();
    const bool aw = top_->s_axil_awvalid_i != 0 && top_->s_axil_awready_o != 0;
    const bool w = top_->s_axil_wvalid_i != 0 && top_->s_axil_wready_o != 0;
    const bool b = top_->s_axil_bvalid_o != 0;
    Edge();
    if (aw) {
      top_->s_axil_awvalid_i = 0;
    }
    if (w) {
      top_->s_axil_wvalid_i = 0;
    }
    if (b) {
      top_->s_axil_bready_i = 0;
      return;
    }
  }
}

uint32_t NpuBench::ReadReg(uint32_t offset) {
  top_->s_axil_araddr_i = offset;
  top_->s_axil_arvalid_i = 1;
  top_->s_axil_rready_i = 1;
  uint32_t value = 0;
  for (int i = 0; i < kHandshakeLimit; ++i) {
    Settle();
    const bool ar = top_->s_axil_arvalid_i != 0 && top_->s_axil_arready_o != 0;
    const bool r = top_->s_axil_rvalid_o != 0;
    value = top_->s_axil_rdata_o;
    Edge();
    if (ar) {
      top_->s_axil_arvalid_i = 0;
    }
    if (r) {
      top_->s_axil_rready_i = 0;
      return value;
    }
  }
  return value;
}

void NpuBench::LoadProgram(std::span<const Instruction> program,
                           uint32_t address) {
  for (const Instruction& instruction : program) {
    for (uint32_t i = 0; i < 4; ++i) {
      ddr_.Write32(address + (4 * i), instruction.words.at(i));
    }
    address += 16;
  }
}

void NpuBench::Start(uint32_t address, uint32_t count) {
  WriteReg(kRegInstrBase, address);
  WriteReg(kRegInstrCount, count);
  WriteReg(kRegCtrl, 1);
}

bool NpuBench::WaitStatus(const std::function<bool(uint32_t)>& done,
                          uint64_t max_cycles) {
  const uint64_t limit = cycle_ + max_cycles;
  while (cycle_ < limit) {
    if (done(ReadReg(kRegStatus))) {
      return true;
    }
    for (int i = 0; i < 64; ++i) {
      Tick();
    }
  }
  return false;
}

bool NpuBench::Run(std::span<const Instruction> program, uint64_t max_cycles) {
  LoadProgram(program, kProgramBase);
  Start(kProgramBase, static_cast<uint32_t>(program.size()));
  return WaitStatus(
      [](uint32_t status) {
        return (status & (kStatusHalted | kStatusError)) != 0;
      },
      max_cycles);
}

}  // namespace hotarubi::tb
