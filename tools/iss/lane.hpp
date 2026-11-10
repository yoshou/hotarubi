#ifndef HOTARUBI_TOOLS_ISS_LANE_HPP_
#define HOTARUBI_TOOLS_ISS_LANE_HPP_

#include <array>
#include <cstdint>
#include <span>

#include "hotarubi/isa.hpp"
#include "memories.hpp"
#include "word.hpp"

namespace hotarubi::iss {

using isa::pass::B;
using isa::pass::C0;
using isa::pass::Cmp;
using isa::pass::CmpLeft;
using isa::pass::Ialu;
using isa::pass::Map;
using isa::pass::Nl;
using isa::pass::Pbuf;
using isa::pass::Post;
using isa::pass::Reduce;
using isa::pass::RowMask;
using isa::pass::ScalarFn;
using isa::pass::Sel;
using isa::pass::Write;

struct Pass {
  uint32_t rx = 0;
  uint32_t rx2 = 0;
  uint32_t rd = 0;
  uint32_t rq = 0;
  uint32_t rnw = 0;
  uint32_t rn8 = 0;
  uint32_t rpb = 0;
  uint32_t rimm = 0;
  uint32_t rimm2 = 0;
  bool x2_en = false;
  uint32_t mg = 0;
  uint32_t ms = 0;
  Map mx = Map::kNone;
  Map mx2 = Map::kNone;
  Map md = Map::kNone;
  Pbuf pa = Pbuf::kNone;
  C0 c0 = C0::kZero;
  B b = B::kOne;
  Nl nl = Nl::kNone;
  Post po = Post::kAdd;
  bool pv = false;
  Reduce red = Reduce::kNone;
  Write wr = Write::kNone;
  ScalarFn sf = ScalarFn::kNone;
  bool st = false;
  bool srn = false;
  bool seps = false;
  uint32_t px = 0;
  Cmp cmp = Cmp::kNone;
  CmpLeft cl = CmpLeft::kX;
  bool cr = false;
  bool cmp_int = false;
  Sel sel = Sel::kNone;
  Ialu ialu = Ialu::kNone;
  bool dst_s1 = false;
  bool c0_pred = false;
  bool x_s1 = false;
  uint32_t ctx = 0;
  RowMask rm = RowMask::kNone;
};

Pass DecodePass(const Word128& instruction);

uint32_t WordOf(const Pass& pass, Map mode, uint32_t w);
uint32_t PermutedIndex(uint32_t px, uint32_t t, uint32_t l);

struct RowState {
  std::array<uint32_t, kTokens> rs0{};
  std::array<uint32_t, kTokens> rs1{};
};

struct LaneInputs {
  uint32_t x = 0;
  uint32_t x2 = 0;
  uint32_t pbuf = 0;
  uint32_t t = 0;
  uint32_t l = 0;
  uint32_t w = 0;
  uint32_t imm = 0;
  uint32_t imm2 = 0;
};

uint32_t LaneOutput(const Pass& pass, const LaneInputs& in,
                    const RowState& rows, std::span<const uint64_t> tbuf);

bool Predicate(const Pass& pass, const LaneInputs& in);
uint32_t PostAdd(uint32_t prod, uint32_t c1, Post op);
uint32_t Interpolate(uint64_t entry, uint32_t fraction);
uint32_t Activation(uint32_t prod, std::span<const uint64_t> tbuf);
uint32_t Exp2(uint32_t prod, std::span<const uint64_t> tbuf);
uint32_t Table01(uint32_t prod, std::span<const uint64_t> tbuf);
uint32_t MaxKey(uint32_t bits);
uint32_t Fp32Max(uint32_t a, uint32_t b);
uint8_t QuantizeElement(uint32_t value, uint32_t emax);
AbufWord QuantizeBlock(std::span<const FbufWord, 4> words);

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_LANE_HPP_
