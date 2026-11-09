#include "vec_engine_bench.hpp"

#include <Vvec_engine.h>

#include <array>
#include <cstdint>
#include <memory>
#include <random>

#include "hotarubi/require.hpp"
#include "memories.hpp"
#include "vec_engine.hpp"
#include "word.hpp"

namespace hotarubi::tb {

struct VecEngineBench::State {
  iss::FbufWord fbuf_q1{};
  iss::FbufWord fbuf_q2{};
  iss::FbufWord pbuf_q1{};
  iss::FbufWord pbuf_q{};
  std::array<uint64_t, iss::kLanes> tab_r{};
  iss::Word128 cbuf_q1{};
  iss::Word128 cbuf_q{};
  bool abuf_we = false;
  uint32_t abuf_addr = 0;
  std::array<uint32_t, 33> abuf_data{};
  bool fbuf_we = false;
  uint32_t fbuf_addr = 0;
  iss::FbufWord fbuf_data{};
};

namespace {

constexpr uint32_t kResetCycles = 4;
constexpr uint64_t kMaxCycles = 50'000'000;

void Drive(Vvec_engine& model, const VecEngineBench::State& state) {
  for (uint32_t i = 0; i < 32; ++i) {
    model.fbuf_rdata_i[i] = state.fbuf_q2.at(i);
    model.pbuf_a_rdata_i[i] = state.pbuf_q.at(i);
  }
  for (uint32_t l = 0; l < iss::kLanes; ++l) {
    model.tab_data_i[l] = state.tab_r.at(l);
  }
  model.cbuf_rdata_i[0] = static_cast<uint32_t>(state.cbuf_q.low);
  model.cbuf_rdata_i[1] = static_cast<uint32_t>(state.cbuf_q.low >> 32);
  model.cbuf_rdata_i[2] = static_cast<uint32_t>(state.cbuf_q.high);
  model.cbuf_rdata_i[3] = static_cast<uint32_t>(state.cbuf_q.high >> 32);
}

iss::AbufWord ToAbufWord(const std::array<uint32_t, 33>& data) {
  iss::AbufWord word{};
  for (uint32_t i = 0; i < 33; ++i) {
    for (uint32_t b = 0; b < 4; ++b) {
      word.at((i * 4) + b) = static_cast<uint8_t>(data.at(i) >> (8 * b));
    }
  }
  return word;
}

}  // namespace

VecEngineBench::VecEngineBench(uint32_t deny_percent, uint32_t seed)
    : model_(std::make_unique<Vvec_engine>()),
      state_(std::make_unique<State>()),
      generator_(seed),
      deny_percent_(deny_percent) {
  Vvec_engine& model = *model_;
  model.clk_i = 0;
  model.rst_ni = 0;
  model.cmd_valid_i = 0;
  model.fbuf_rd_grant_i = 0;
  model.fbuf_wr_grant_i = 0;
  model.abuf_wr_grant_i = 0;
  Drive(model, *state_);
  for (uint32_t i = 0; i < kResetCycles; ++i) {
    model.clk_i = 0;
    model.eval();
    model.clk_i = 1;
    model.eval();
  }
  model.rst_ni = 1;
}

VecEngineBench::~VecEngineBench() { model_->final(); }

void VecEngineBench::Tick(iss::Memories& memories) {
  Vvec_engine& model = *model_;
  State& state = *state_;
  const auto deny = [&] {
    return deny_percent_ != 0 && std::uniform_int_distribution<uint32_t>(
                                     0, 99)(generator_) < deny_percent_;
  };
  model.clk_i = 0;
  Drive(model, state);
  model.eval();
  model.fbuf_rd_grant_i = (model.fbuf_rd_req_o != 0 && !deny()) ? 1 : 0;
  model.fbuf_wr_grant_i = (model.fbuf_wr_req_o != 0 && !deny()) ? 1 : 0;
  model.abuf_wr_grant_i = model.abuf_wr_req_o;
  model.eval();

  State next = state;
  next.fbuf_q1 = memories.fbuf.at(model.fbuf_raddr_o & (iss::kFbufDepth - 1));
  next.fbuf_q2 = state.fbuf_q1;
  next.pbuf_q1 = memories.pbuf.at(model.pbuf_a_addr_o & (iss::kPbufDepth - 1));
  next.pbuf_q = state.pbuf_q1;
  for (uint32_t l = 0; l < iss::kLanes; ++l) {
    next.tab_r.at(l) =
        memories.tbuf.at(model.tab_addr_o[l] & (iss::kTbufDepth - 1));
  }
  next.cbuf_q1 = memories.cbuf.at(model.cbuf_addr_o & (iss::kCbufDepth - 1));
  next.cbuf_q = state.cbuf_q1;
  if (state.abuf_we) {
    memories.abuf.at(state.abuf_addr) = ToAbufWord(state.abuf_data);
  }
  next.abuf_we = model.abuf_wr_req_o != 0;
  if (next.abuf_we) {
    next.abuf_addr = model.abuf_waddr_o & (iss::kAbufDepth - 1);
    for (uint32_t i = 0; i < 33; ++i) {
      next.abuf_data.at(i) = model.abuf_wdata_o[i];
    }
  }
  if (state.fbuf_we) {
    memories.fbuf.at(state.fbuf_addr) = state.fbuf_data;
  }
  next.fbuf_we = model.fbuf_wr_grant_i != 0;
  if (next.fbuf_we) {
    next.fbuf_addr = model.fbuf_waddr_o & (iss::kFbufDepth - 1);
    for (uint32_t i = 0; i < 32; ++i) {
      next.fbuf_data.at(i) =
          model.fbuf_wtr_o != 0 ? model.fbuf_tdata_o[i] : model.fbuf_wdata_o[i];
    }
  }

  model.clk_i = 1;
  model.eval();
  state = next;
}

VecEngineRun VecEngineBench::Run(const iss::VecCommand& command,
                                 iss::Memories& memories) {
  Vvec_engine& model = *model_;
  model.cmd_valid_i = 1;
  model.cmd_op_i = command.op & 0x1FU;
  model.cmd_q_i = command.quantize ? 1 : 0;
  model.cmd_src_i = command.src & 0xFFFU;
  model.cmd_src2_i = command.src2 & 0xFFFU;
  model.cmd_dst_i = command.dst & 0x1FFFU;
  model.cmd_rows_q_i = command.rows_q & 0x1FFU;
  model.cmd_n8_i = command.n8 & 0xFFU;
  model.cmd_p_base_i = command.p_base & 0x3FFU;
  model.cmd_aux_i = command.aux & 0x7FFU;
  model.cmd_imm_i = command.imm;
  model.clk_i = 0;
  model.eval();
  Require(model.cmd_ready_o != 0, "the VEC engine is not idle");
  Tick(memories);
  model.cmd_valid_i = 0;
  VecEngineRun run;
  while (true) {
    model.clk_i = 0;
    Drive(model, *state_);
    model.eval();
    ++run.cycles;
    if (model.busy_o == 0) {
      break;
    }
    Require(run.cycles < kMaxCycles, "the VEC command does not complete");
    run.issue_cycles += model.stat_issue_o;
    run.drain_cycles += model.stat_drain_o;
    run.scalar_cycles += model.stat_scalar_o;
    run.passes += model.stat_pass_o;
    Tick(memories);
  }
  for (uint32_t i = 0; i < 2; ++i) {
    Tick(memories);
  }
  return run;
}

}  // namespace hotarubi::tb
