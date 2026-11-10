#include "vec_rom.hpp"

#include <array>
#include <cstdint>
#include <format>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "hotarubi/isa.hpp"
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
constexpr uint32_t kQ = 9;
constexpr uint32_t kOne = 11;
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
  uint32_t scratch = kImm;
  uint32_t dst = kDst;
  uint32_t ctx = 0;
};

Slot SlotOf(uint32_t index, uint32_t scratch) {
  if (index == 0) {
    return {.src = kSrc, .scratch = scratch, .dst = kDst, .ctx = 0};
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

void Grouped(vp::Assembler& a, uint32_t scratch,
             const std::vector<QuadPass>& passes) {
  constexpr uint32_t kShift = kContexts == 4 ? 2 : 1;
  a.Alu(vp::Salu::kShr, kGroups, kRowsQ, 0, kShift);
  a.Alu(vp::Salu::kAnd, kTail, kRowsQ, 0, kContexts - 1);
  a.Alu(vp::Salu::kShl, kGroupN8, kN8, 0, kShift);
  a.Alu(vp::Salu::kShl, kGroupDstStep, kDstStep, 0, kShift);
  for (uint32_t k = 1; k < kContexts; ++k) {
    const Slot previous = SlotOf(k - 1, scratch);
    const Slot slot = SlotOf(k, scratch);
    a.Alu(vp::Salu::kAdd, slot.src, previous.src, kN8);
    a.Alu(vp::Salu::kAdd, slot.scratch, previous.scratch, kN8);
    a.Alu(vp::Salu::kAdd, slot.dst, previous.dst, kDstStep);
  }
  a.Branch(vp::Cond::kZ, kGroups, "tail");
  a.Label("group");
  for (const QuadPass& pass : passes) {
    for (uint32_t k = 0; k < kContexts; ++k) {
      a.Emit(pass(SlotOf(k, scratch)));
    }
  }
  for (uint32_t k = 0; k < kContexts; ++k) {
    const Slot slot = SlotOf(k, scratch);
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
    a.Emit(pass(SlotOf(0, scratch)));
  }
  a.Alu(vp::Salu::kAdd, kSrc, kSrc, kN8);
  a.Alu(vp::Salu::kAdd, scratch, scratch, kN8);
  a.Alu(vp::Salu::kAdd, kDst, kDst, kDstStep);
  a.Alu(vp::Salu::kSub, kTail, kTail, 0, 1);
  a.Branch(vp::Cond::kNz, kTail, "single");
  a.Label("end");
  a.Ret();
}

void LayerNorm(vp::Assembler& a) {
  QuadPrologue(a);
  a.Memory(vp::Smem::kRn, {.ra = kAux});
  a.Alu(vp::Salu::kAdd, kBeta, kPBase, kQuarterN8);
  Grouped(a, kSrc2,
          {[](const Slot& s) {
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
  a.Memory(vp::Smem::kRn, {.ra = kAux});
  Grouped(a, kSrc2,
          {[](const Slot& s) {
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
  sigma.rd = kImm;
  sigma.nl = vp::Nl::kAct;
  sigma.wr = vp::Write::kFbuf;
  a.Emit(sigma);
  vp::Pass gate = Whole();
  gate.rx2 = kImm;
  gate.b = vp::B::kX2;
  gate.post = vp::Post::kSignfix;
  if (multiply) {
    gate.rd = kImm;
    gate.wr = vp::Write::kFbuf;
    a.Emit(gate);
    vp::Pass product = Whole();
    product.rx = kImm;
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
  Grouped(a, kImm,
          {[](const Slot& s) {
             vp::Pass maximum = Quad(s);
             maximum.rimm2 = kAux;
             maximum.red = vp::Reduce::kMax;
             maximum.sc = vp::kNegMax;
             maximum.cmp = vp::Cmp::kLt;
             maximum.cmp_left = vp::CmpLeft::kRowColumn;
             maximum.sel = vp::Sel::kNull;
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
             exponent.red = vp::Reduce::kSum;
             exponent.wr = vp::Write::kFbuf;
             exponent.sc = vp::kRecipSum;
             exponent.cmp = vp::Cmp::kLt;
             exponent.cmp_left = vp::CmpLeft::kRowColumn;
             exponent.sel = vp::Sel::kZero;
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
    pass.rd = kImm;
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
    pass.rx2 = kImm;
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

void Program(uint32_t entry, vp::Assembler& a) {
  namespace lib = isa::library;
  vp::Pass pass = Whole();
  switch (entry) {
    case lib::kAdd:
      pass.rx2 = kSrc2;
      pass.c0 = vp::C0::kX2;
      break;
    case lib::kMul:
      pass.rx2 = kSrc2;
      pass.b = vp::B::kX2;
      break;
    case lib::kMax:
      pass.rx2 = kSrc2;
      pass.post = vp::Post::kMax;
      pass.post_x2 = true;
      break;
    case lib::kAddP:
      pass.rpb = kPBase;
      pass.pa = vp::Pbuf::kRowVec;
      pass.c0 = vp::C0::kPbuf;
      break;
    case lib::kMulP:
      pass.rpb = kPBase;
      pass.pa = vp::Pbuf::kRowVec;
      pass.b = vp::B::kPbuf;
      break;
    case lib::kScale:
      pass.rimm = kImm;
      pass.b = vp::B::kImm;
      break;
    case lib::kCopy:
      break;
    case lib::kRelu:
      pass.post = vp::Post::kMax;
      break;
    case lib::kAct:
      Activation(a, false);
      return;
    case lib::kActMul:
      Activation(a, true);
      return;
    case lib::kRope:
      Rope(a);
      return;
    case lib::kSoftmax:
      Softmax(a);
      return;
    case lib::kLayerNorm:
      LayerNorm(a);
      return;
    case lib::kRmsNorm:
      RmsNorm(a);
      return;
    default:
      Fail(std::format("no library program at entry {}", entry));
  }
  Elementwise(a, pass);
}

}  // namespace

Rom Build() {
  Rom rom;
  uint32_t end = 0;
  for (const isa::library::Program& program : isa::library::kPrograms) {
    const uint32_t start = program.entry * isa::vec::kEntryWords;
    Require(start >= end,
            std::format("library program {} at word {} overlaps the one "
                        "before, which ends at word {}",
                        program.name, start, end));
    vp::Assembler a;
    Program(program.entry, a);
    uint32_t pc = start;
    for (const Instruction& word : a.Words(start)) {
      Require(pc < vp::kRomWords, "the VEC library exceeds the ROM");
      rom.words.at(pc++) = word;
    }
    end = pc;
  }
  return rom;
}

std::string Package(const Rom& rom) {
  std::string text =
      "package vec_rom_pkg;\n\n  localparam logic [127:0] Rom [256] = '{\n";
  for (uint32_t i = 0; i < vp::kRomWords; ++i) {
    text += std::format("    128'h{}{}\n", rom.words.at(i).Hex(),
                        i + 1 == vp::kRomWords ? "" : ",");
  }
  text += "  };\n\nendpackage\n";
  return text;
}

}  // namespace hotarubi::compiler::vec_rom
