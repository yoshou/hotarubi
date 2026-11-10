#include "vec_rom.hpp"

#include <array>
#include <cstdint>
#include <format>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "hotarubi/require.hpp"
#include "isa.hpp"
#include "vec_engine.hpp"
#include "vprog.hpp"

namespace hotarubi::compiler::vec_rom {
namespace {

namespace vp = vprog;

constexpr uint32_t kSrc = 1;
constexpr uint32_t kSrc2 = 2;
constexpr uint32_t kDst = 3;
constexpr uint32_t kRowsQ = 4;
constexpr uint32_t kN8 = 5;
constexpr uint32_t kPBase = 6;
constexpr uint32_t kAux = 7;
constexpr uint32_t kImm = 8;
constexpr uint32_t kScratch = 9;
constexpr uint32_t kQ = 10;
constexpr uint32_t kOne = 11;
constexpr uint32_t kValid = 12;
constexpr uint32_t kCount = 13;
constexpr uint32_t kQuarterN8 = 14;
constexpr uint32_t kBeta = 15;
constexpr uint32_t kDstStep = 16;
constexpr uint32_t kTemp = 17;
constexpr uint32_t kGroups = 18;
constexpr uint32_t kHeadWords = 11;
constexpr uint32_t kTilePbuf = 12;
constexpr uint32_t kTail = 19;
constexpr uint32_t kGroupN8 = 20;
constexpr uint32_t kGroupDstStep = 21;
constexpr uint32_t kSlotBase = 22;
constexpr uint32_t kLog2eRegister = 25;
constexpr uint32_t kContexts = iss::kContexts;

struct Slot {
  uint32_t src = kSrc;
  uint32_t scratch = kScratch;
  uint32_t dst = kDst;
  uint32_t ctx = 0;
};

Slot SlotOf(uint32_t index) {
  if (index == 0) {
    return {};
  }
  const uint32_t base = kSlotBase + (3 * (index - 1));
  return {.src = base, .scratch = base + 1, .dst = base + 2, .ctx = index};
}

using QuadPass = std::function<vp::Pass(const Slot&)>;

vp::Pass Whole() {
  return {.rx = kSrc,
          .rd = kDst,
          .rq = kRowsQ,
          .rnw = kN8,
          .rn8 = kN8,
          .wr = vp::Write::kFinal};
}

vp::Pass Quad(const Slot& slot) {
  return {.rx = slot.src, .rq = kOne, .rnw = kN8, .rn8 = kN8, .ctx = slot.ctx};
}

void Elementwise(vp::Assembler& a, vp::Pass pass) {
  a.Emit(pass);
  a.Ret();
}

void QuadPrologue(vp::Assembler& a) {
  a.LoadImmediate(kOne, 1);
  a.Alu(vp::Salu::kShr, kQuarterN8, kN8, 0, 2);
  a.Alu(vp::Salu::kSub, kTemp, kN8, kQuarterN8);
  a.Alu(vp::Salu::kMul, kTemp, kTemp, kQ);
  a.Alu(vp::Salu::kSub, kDstStep, kN8, kTemp);
}

void CountPrologue(vp::Assembler& a) {
  a.Alu(vp::Salu::kSeq, kValid, kAux, 0, 0);
  a.Alu(vp::Salu::kShl, kCount, kN8, 0, 3);
  a.Alu(vp::Salu::kSub, kCount, kCount, kAux);
  a.Alu(vp::Salu::kMul, kCount, kCount, kValid);
  a.Alu(vp::Salu::kAdd, kCount, kCount, kAux);
  a.Memory(vp::Smem::kRn, {.ra = kCount});
}

void Grouped(vp::Assembler& a, const std::vector<QuadPass>& passes) {
  constexpr uint32_t kShift = kContexts == 4 ? 2 : 1;
  a.Alu(vp::Salu::kShr, kGroups, kRowsQ, 0, kShift);
  a.Alu(vp::Salu::kAnd, kTail, kRowsQ, 0, kContexts - 1);
  a.Alu(vp::Salu::kShl, kGroupN8, kN8, 0, kShift);
  a.Alu(vp::Salu::kShl, kGroupDstStep, kDstStep, 0, kShift);
  for (uint32_t k = 1; k < kContexts; ++k) {
    const Slot previous = SlotOf(k - 1);
    const Slot slot = SlotOf(k);
    a.Alu(vp::Salu::kAdd, slot.src, previous.src, kN8);
    a.Alu(vp::Salu::kAdd, slot.scratch, previous.scratch, kN8);
    a.Alu(vp::Salu::kAdd, slot.dst, previous.dst, kDstStep);
  }
  a.Branch(vp::Cond::kZ, kGroups, "tail");
  a.Label("group");
  for (const QuadPass& pass : passes) {
    for (uint32_t k = 0; k < kContexts; ++k) {
      a.Emit(pass(SlotOf(k)));
    }
  }
  for (uint32_t k = 0; k < kContexts; ++k) {
    const Slot slot = SlotOf(k);
    a.Alu(vp::Salu::kAdd, slot.src, slot.src, kGroupN8);
    a.Alu(vp::Salu::kAdd, slot.scratch, slot.scratch, kGroupN8);
    a.Alu(vp::Salu::kAdd, slot.dst, slot.dst, kGroupDstStep);
  }
  a.Alu(vp::Salu::kSub, kGroups, kGroups, 0, 1);
  a.Branch(vp::Cond::kNz, kGroups, "group");
  a.Label("tail");
  a.Branch(vp::Cond::kZ, kTail, "end");
  a.Label("single");
  for (const QuadPass& pass : passes) {
    a.Emit(pass(SlotOf(0)));
  }
  a.Alu(vp::Salu::kAdd, kSrc, kSrc, kN8);
  a.Alu(vp::Salu::kAdd, kScratch, kScratch, kN8);
  a.Alu(vp::Salu::kAdd, kDst, kDst, kDstStep);
  a.Alu(vp::Salu::kSub, kTail, kTail, 0, 1);
  a.Branch(vp::Cond::kNz, kTail, "single");
  a.Label("end");
  a.Ret();
}

void LayerNorm(vp::Assembler& a) {
  QuadPrologue(a);
  CountPrologue(a);
  a.Alu(vp::Salu::kAdd, kBeta, kPBase, kQuarterN8);
  Grouped(a, {[](const Slot& s) {
                vp::Pass sum = Quad(s);
                sum.red = vp::Reduce::kSum;
                sum.sc = vp::kMean;
                return sum;
              },
              [](const Slot& s) {
                vp::Pass squares = Quad(s);
                squares.rimm = kImm;
                squares.c0 = vp::C0::kRs0;
                squares.b = vp::B::kPre;
                squares.red = vp::Reduce::kSum;
                squares.sc = vp::kRstd;
                return squares;
              },
              [](const Slot& s) {
                vp::Pass normalize = Quad(s);
                normalize.rd = s.scratch;
                normalize.c0 = vp::C0::kRs0;
                normalize.b = vp::B::kRs1;
                normalize.wr = vp::Write::kFbuf;
                return normalize;
              },
              [](const Slot& s) {
                vp::Pass gamma = Quad(s);
                gamma.rx = s.scratch;
                gamma.rd = s.scratch;
                gamma.rpb = kPBase;
                gamma.pa = vp::Pbuf::kRowVec;
                gamma.b = vp::B::kPbuf;
                gamma.wr = vp::Write::kFbuf;
                return gamma;
              },
              [](const Slot& s) {
                vp::Pass beta = Quad(s);
                beta.rx = s.scratch;
                beta.rd = s.dst;
                beta.rpb = kBeta;
                beta.pa = vp::Pbuf::kRowVec;
                beta.c0 = vp::C0::kPbuf;
                beta.wr = vp::Write::kFinal;
                return beta;
              }});
}

void RmsNorm(vp::Assembler& a) {
  QuadPrologue(a);
  CountPrologue(a);
  Grouped(a, {[](const Slot& s) {
                vp::Pass squares = Quad(s);
                squares.rimm = kImm;
                squares.b = vp::B::kPre;
                squares.red = vp::Reduce::kSum;
                squares.sc = vp::kRstd;
                return squares;
              },
              [](const Slot& s) {
                vp::Pass normalize = Quad(s);
                normalize.rd = s.scratch;
                normalize.b = vp::B::kRs1;
                normalize.wr = vp::Write::kFbuf;
                return normalize;
              },
              [](const Slot& s) {
                vp::Pass gamma = Quad(s);
                gamma.rx = s.scratch;
                gamma.rd = s.dst;
                gamma.rpb = kPBase;
                gamma.pa = vp::Pbuf::kRowVec;
                gamma.b = vp::B::kPbuf;
                gamma.wr = vp::Write::kFinal;
                return gamma;
              }});
}

void Activation(vp::Assembler& a, bool multiply) {
  vp::Pass sigma = Whole();
  sigma.rd = kScratch;
  sigma.nl = vp::Nl::kAct;
  sigma.wr = vp::Write::kFbuf;
  a.Emit(sigma);
  vp::Pass gate = Whole();
  gate.rx2 = kScratch;
  gate.b = vp::B::kX2;
  gate.post = vp::Post::kSignfix;
  if (multiply) {
    gate.rd = kScratch;
    gate.wr = vp::Write::kFbuf;
    a.Emit(gate);
    vp::Pass product = Whole();
    product.rx = kScratch;
    product.rx2 = kSrc2;
    product.b = vp::B::kX2;
    a.Emit(product);
  } else {
    a.Emit(gate);
  }
  a.Ret();
}

void Softmax(vp::Assembler& a) {
  a.LoadImmediate(kLog2eRegister, vp::kLog2e);
  QuadPrologue(a);
  Grouped(a, {[](const Slot& s) {
                vp::Pass maximum = Quad(s);
                maximum.rimm2 = kAux;
                maximum.mask_en = true;
                maximum.red = vp::Reduce::kMax;
                maximum.sc = vp::kNegMax;
                return maximum;
              },
              [](const Slot& s) {
                vp::Pass exponent = Quad(s);
                exponent.rd = s.scratch;
                exponent.rimm = kLog2eRegister;
                exponent.rimm2 = kAux;
                exponent.c0 = vp::C0::kRs0;
                exponent.b = vp::B::kImm;
                exponent.nl = vp::Nl::kExp2;
                exponent.mask_en = true;
                exponent.red = vp::Reduce::kSum;
                exponent.wr = vp::Write::kFbuf;
                exponent.sc = vp::kRecipSum;
                return exponent;
              },
              [](const Slot& s) {
                vp::Pass normalize = Quad(s);
                normalize.rx = s.scratch;
                normalize.rd = s.dst;
                normalize.b = vp::B::kRs1;
                normalize.wr = vp::Write::kFinal;
                return normalize;
              }});
}

void Rope(vp::Assembler& a) {
  a.Alu(vp::Salu::kShl, kHeadWords, kAux, 0, 2);
  a.Alu(vp::Salu::kShl, kTilePbuf, kRowsQ, 0, 2);
  a.Alu(vp::Salu::kAdd, kTilePbuf, kTilePbuf, kPBase);
  const auto half = [](vp::Map x, vp::Map d) {
    vp::Pass pass = Whole();
    pass.rd = kScratch;
    pass.rnw = kHeadWords;
    pass.rpb = kTilePbuf;
    pass.map = {
        .group = 2, .stride = 3, .x = x, .x2 = vp::Map::kNone, .dst = d};
    pass.pa = vp::Pbuf::kTile;
    pass.b = vp::B::kPbuf;
    pass.wr = vp::Write::kFbuf;
    return pass;
  };
  const auto combine = [](vp::Map x, vp::Map d, bool negate) {
    vp::Pass pass = Whole();
    pass.rx2 = kScratch;
    pass.rnw = kHeadWords;
    pass.rpb = kPBase;
    pass.map = {.group = 2, .stride = 3, .x = x, .x2 = d, .dst = d};
    pass.pa = vp::Pbuf::kTile;
    pass.b = vp::B::kPbuf;
    pass.post = negate ? vp::Post::kSub : vp::Post::kAdd;
    pass.post_x2 = true;
    return pass;
  };
  a.Emit(half(vp::Map::kUpper, vp::Map::kGroup));
  a.Emit(combine(vp::Map::kGroup, vp::Map::kGroup, true));
  a.Emit(half(vp::Map::kGroup, vp::Map::kUpper));
  a.Emit(combine(vp::Map::kUpper, vp::Map::kUpper, false));
  a.Ret();
}

void Program(uint32_t op, vp::Assembler& a) {
  switch (op) {
    case 0: {
      vp::Pass pass = Whole();
      pass.rx2 = kSrc2;
      pass.c0 = vp::C0::kX2;
      Elementwise(a, pass);
      break;
    }
    case 1: {
      vp::Pass pass = Whole();
      pass.rx2 = kSrc2;
      pass.b = vp::B::kX2;
      Elementwise(a, pass);
      break;
    }
    case 2: {
      vp::Pass pass = Whole();
      pass.rpb = kPBase;
      pass.pa = vp::Pbuf::kRowVec;
      pass.c0 = vp::C0::kPbuf;
      Elementwise(a, pass);
      break;
    }
    case 3: {
      vp::Pass pass = Whole();
      pass.rpb = kPBase;
      pass.pa = vp::Pbuf::kRowVec;
      pass.b = vp::B::kPbuf;
      Elementwise(a, pass);
      break;
    }
    case 4: {
      vp::Pass pass = Whole();
      pass.rimm = kImm;
      pass.b = vp::B::kImm;
      Elementwise(a, pass);
      break;
    }
    case 5:
      Elementwise(a, Whole());
      break;
    case 6:
      LayerNorm(a);
      break;
    case 7:
      RmsNorm(a);
      break;
    case 8:
    case 9:
      Activation(a, false);
      break;
    case 10:
      Activation(a, true);
      break;
    case 11:
      Softmax(a);
      break;
    case 12:
      Rope(a);
      break;
    case 13: {
      vp::Pass pass = Whole();
      pass.wr = vp::Write::kAbuf;
      Elementwise(a, pass);
      break;
    }
    case 14: {
      vp::Pass pass = Whole();
      pass.post = vp::Post::kMax;
      Elementwise(a, pass);
      break;
    }
    default: {
      vp::Pass pass = Whole();
      pass.rx2 = kSrc2;
      pass.post = vp::Post::kMax;
      pass.post_x2 = true;
      Elementwise(a, pass);
      break;
    }
  }
}

}  // namespace

Rom Build() {
  Rom rom;
  uint32_t pc = 0;
  for (uint32_t op = 0; op < kOperations; ++op) {
    vp::Assembler a;
    Program(op, a);
    rom.starts.at(op) = pc;
    for (const Instruction& word : a.Words(pc)) {
      Require(pc < vp::kRomWords, "the VEC ROM programs exceed 256 words");
      rom.words.at(pc++) = word;
    }
  }
  return rom;
}

std::string Package(const Rom& rom) {
  std::string text =
      "package vec_rom_pkg;\n\n  localparam logic [9:0] RomPc [16] = '{\n";
  for (uint32_t op = 0; op < kOperations; ++op) {
    std::string_view separator = ",";
    if (op + 1 == kOperations) {
      separator = "\n";
    } else if (op % 8 == 7) {
      separator = ",\n";
    }
    text += std::format("{}10'd{}{}", op % 8 == 0 ? "    " : " ",
                        rom.starts.at(op), separator);
  }
  text += "  };\n\n  localparam logic [127:0] Rom [256] = '{\n";
  for (uint32_t i = 0; i < vp::kRomWords; ++i) {
    text += std::format("    128'h{}{}\n", rom.words.at(i).Hex(),
                        i + 1 == vp::kRomWords ? "" : ",");
  }
  text += "  };\n\nendpackage\n";
  return text;
}

}  // namespace hotarubi::compiler::vec_rom
