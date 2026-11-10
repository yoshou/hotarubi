#include "vprog.hpp"

#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "hotarubi/isa.hpp"
#include "isa.hpp"

namespace hotarubi::compiler::vprog {
namespace {

namespace pg = isa::program;

template <typename E>
uint32_t U(E value) {
  return static_cast<uint32_t>(value);
}

uint32_t U(bool value) { return value ? 1 : 0; }

bool Arithmetic(const Pass& pass) {
  return pass.c0 != C0::kZero || pass.b != B::kOne || pass.nl != Nl::kNone ||
         pass.post != Post::kAdd || pass.post_x2 || pass.ialu != Ialu::kNone ||
         pass.red != Reduce::kNone || pass.pa != Pbuf::kNone;
}

}  // namespace

Instruction EncodePass(const Pass& pass) {
  namespace f = isa::pass;
  Instruction instruction;
  instruction.Set(pg::kKind, pg::kKindPass)
      .Set(f::kRx, pass.rx)
      .Set(f::kRx2, pass.rx2)
      .Set(f::kRd, pass.rd)
      .Set(f::kRq, pass.rq)
      .Set(f::kRnw, pass.rnw)
      .Set(f::kRn8, pass.rn8)
      .Set(f::kRpb, pass.rpb)
      .Set(f::kRimm, pass.rimm)
      .Set(f::kRimm2, pass.rimm2)
      .Set(f::kX2En, U(pass.x2_en))
      .Set(f::kMg, pass.map.group)
      .Set(f::kMs, pass.map.stride)
      .Set(f::kMx, U(pass.map.x))
      .Set(f::kMx2, U(pass.map.x2))
      .Set(f::kMd, U(pass.map.dst))
      .Set(f::kPa, U(pass.pa))
      .Set(f::kC0, U(pass.c0))
      .Set(f::kB, U(pass.b))
      .Set(f::kNl, U(pass.nl))
      .Set(f::kPo, U(pass.post))
      .Set(f::kPv, U(pass.post_x2))
      .Set(f::kRed, U(pass.red))
      .Set(f::kWr, U(pass.wr))
      .Set(f::kSf, U(pass.sc.fn))
      .Set(f::kSt, U(pass.sc.rs0))
      .Set(f::kSrn, U(pass.sc.rn))
      .Set(f::kSeps, U(pass.sc.eps))
      .Set(f::kPx, pass.px)
      .Set(f::kCmp, U(pass.cmp))
      .Set(f::kCl, U(pass.cmp_left))
      .Set(f::kCr, U(pass.cmp_right_x2))
      .Set(f::kCmpInt, U(pass.cmp_int))
      .Set(f::kSel, U(pass.sel))
      .Set(f::kIalu, U(pass.ialu))
      .Set(f::kC0Pred, U(pass.c0_pred))
      .Set(f::kXS1, U(pass.x_s1))
      .Set(f::kDstS1, U(pass.dst_s1))
      .Set(f::kCtx, pass.ctx)
      .Set(f::kRm, U(pass.rm));
  return instruction;
}

Instruction EncodeSalu(Salu op, uint32_t rd, uint32_t ra, uint32_t rb,
                       std::optional<uint32_t> imm) {
  Instruction instruction;
  instruction.Set(pg::kKind, pg::kKindSalu)
      .Set(pg::kOp, U(op))
      .Set(pg::kRd, rd)
      .Set(pg::kRa, ra)
      .Set(pg::kRb, rb);
  if (imm.has_value()) {
    instruction.Set(pg::kUseImm, 1).Set(pg::kImm, *imm);
  }
  return instruction;
}

Instruction EncodeSmem(Smem op, const SmemArgs& args) {
  Instruction instruction;
  instruction.Set(pg::kKind, pg::kKindSmem)
      .Set(pg::kOp, U(op))
      .Set(pg::kRd, args.rd)
      .Set(pg::kRa, args.ra)
      .Set(pg::kRb, args.rb);
  if (args.col.has_value()) {
    instruction.Set(pg::kUseImm, 1).Set(pg::kSmemCol, *args.col);
  }
  instruction.Set(pg::kSmemWhich, U(args.which))
      .Set(pg::kSmemCtx, args.ctx)
      .Set(pg::kImm, args.imm);
  return instruction;
}

Instruction EncodeBranch(Cond cond, uint32_t ra, uint32_t target) {
  Instruction instruction;
  instruction.Set(pg::kKind, pg::kKindBranch)
      .Set(pg::kOp, U(cond))
      .Set(pg::kRa, ra)
      .Set(pg::kImm, target);
  return instruction;
}

Instruction EncodeRet() {
  Instruction instruction;
  instruction.Set(pg::kKind, pg::kKindRet);
  return instruction;
}

Instruction EncodeNop() {
  Instruction instruction;
  instruction.Set(pg::kKind, pg::kKindNop);
  return instruction;
}

void Assembler::Label(std::string_view name) {
  Require(!labels_.contains(name), std::format("duplicate label {}", name));
  labels_.emplace(std::string(name), size_);
}

Assembler& Assembler::Emit(const Pass& pass) {
  const WordMap& map = pass.map;
  const bool upper =
      map.x == Map::kUpper || map.x2 == Map::kUpper || map.dst == Map::kUpper;
  const bool grouped = upper || map.x == Map::kGroup || map.x2 == Map::kGroup ||
                       map.dst == Map::kGroup;
  Require(!grouped || map.stride >= map.group + (upper ? 1 : 0),
          std::format("word map stride {} too small for group {}", map.stride,
                      map.group));
  Pass emitted = pass;
  if (skip_masked_rows_ && emitted.rm == RowMask::kNone) {
    emitted.rm = RowMask::kSkip;
  }
  items_.emplace_back(emitted);
  ++size_;
  return *this;
}

Assembler& Assembler::Alu(Salu op, uint32_t rd, uint32_t ra, uint32_t rb,
                          std::optional<uint32_t> imm) {
  items_.emplace_back(EncodeSalu(op, rd, ra, rb, imm));
  ++size_;
  return *this;
}

Assembler& Assembler::LoadImmediate(uint32_t rd, uint32_t imm) {
  return Alu(Salu::kMov, rd, 0, 0, imm);
}

Assembler& Assembler::Move(uint32_t rd, uint32_t rb) {
  return Alu(Salu::kMov, rd, 0, rb);
}

Assembler& Assembler::Memory(Smem op, const SmemArgs& args) {
  items_.emplace_back(EncodeSmem(op, args));
  ++size_;
  return *this;
}

Assembler& Assembler::Branch(Cond cond, uint32_t ra, std::string_view label) {
  items_.emplace_back(
      BranchItem{.cond = cond, .ra = ra, .label = std::string(label)});
  ++size_;
  return *this;
}

Assembler& Assembler::Ret() {
  items_.emplace_back(EncodeRet());
  ++size_;
  return *this;
}

Assembler& Assembler::Nop() {
  items_.emplace_back(EncodeNop());
  ++size_;
  return *this;
}

Assembler& Assembler::Transpose8(uint32_t src, uint32_t dst, uint32_t tiles,
                                 uint32_t src_tile_stride,
                                 uint32_t src_word_stride,
                                 uint32_t dst_tile_stride,
                                 uint32_t dst_word_stride, uint32_t rotation) {
  namespace tr = isa::transpose;
  Instruction word;
  word.Set(pg::kKind, pg::kKindTranspose)
      .Set(tr::kRx, src)
      .Set(tr::kRsw, src_word_stride)
      .Set(tr::kRd, dst)
      .Set(tr::kRt, tiles)
      .Set(tr::kRdt, dst_tile_stride)
      .Set(tr::kRst, src_tile_stride)
      .Set(tr::kRdw, dst_word_stride)
      .Set(tr::kRor, rotation);
  items_.emplace_back(word);
  ++size_;
  return *this;
}

Assembler& Assembler::StoreScalar(uint32_t rw, uint32_t rlane, uint32_t rcol,
                                  uint32_t rval, uint32_t r_one) {
  return Emit({.rx = rw,
               .rx2 = rlane,
               .rd = rw,
               .rq = r_one,
               .rnw = r_one,
               .rn8 = r_one,
               .rimm = rval,
               .rimm2 = rcol,
               .wr = Write::kFbuf,
               .cmp = Cmp::kNe,
               .cmp_left = CmpLeft::kX2,
               .cmp_int = true,
               .sel = Sel::kImm});
}

Assembler& Assembler::IntToFloat(uint32_t rx, uint32_t rd, uint32_t rq,
                                 uint32_t rnw, uint32_t rn8, uint32_t r_c0,
                                 uint32_t r_ncf, uint32_t rtmp) {
  Emit({.rx = rx,
        .rd = rtmp,
        .rq = rq,
        .rnw = rnw,
        .rn8 = rn8,
        .rimm = r_c0,
        .c0 = C0::kImm,
        .wr = Write::kFbuf,
        .ialu = Ialu::kAdd});
  return Emit({.rx = rtmp,
               .rd = rd,
               .rq = rq,
               .rnw = rnw,
               .rn8 = rn8,
               .rimm = r_ncf,
               .c0 = C0::kImm,
               .wr = Write::kFbuf});
}

Assembler& Assembler::RoundToInt(uint32_t rx, uint32_t rd, uint32_t rq,
                                 uint32_t rnw, uint32_t rn8, uint32_t r_c0,
                                 uint32_t r_cf, uint32_t rtmp) {
  Emit({.rx = rx,
        .rd = rtmp,
        .rq = rq,
        .rnw = rnw,
        .rn8 = rn8,
        .rimm = r_cf,
        .c0 = C0::kImm,
        .wr = Write::kFbuf});
  return Emit({.rx = rtmp,
               .rd = rd,
               .rq = rq,
               .rnw = rnw,
               .rn8 = rn8,
               .rimm = r_c0,
               .c0 = C0::kImm,
               .wr = Write::kFbuf,
               .ialu = Ialu::kSub});
}

Assembler& Assembler::RoundToIntFloat(uint32_t rx, uint32_t rd, uint32_t rq,
                                      uint32_t rnw, uint32_t rn8, uint32_t r_cf,
                                      uint32_t r_ncf, uint32_t rtmp) {
  Emit({.rx = rx,
        .rd = rtmp,
        .rq = rq,
        .rnw = rnw,
        .rn8 = rn8,
        .rimm = r_cf,
        .c0 = C0::kImm,
        .wr = Write::kFbuf});
  return Emit({.rx = rtmp,
               .rd = rd,
               .rq = rq,
               .rnw = rnw,
               .rn8 = rn8,
               .rimm = r_ncf,
               .c0 = C0::kImm,
               .wr = Write::kFbuf});
}

Assembler& Assembler::FloorFloat(uint32_t rx, uint32_t rd, uint32_t rq,
                                 uint32_t rnw, uint32_t rn8, uint32_t r_cf,
                                 uint32_t r_ncf, uint32_t r_none,
                                 uint32_t rtmp) {
  RoundToIntFloat(rx, rd, rq, rnw, rn8, r_cf, r_ncf, rtmp);
  return Emit({.rx = rd,
               .rx2 = rx,
               .rd = rd,
               .rq = rq,
               .rnw = rnw,
               .rn8 = rn8,
               .rimm = r_none,
               .c0 = C0::kImm,
               .wr = Write::kFbuf,
               .cmp = Cmp::kGt,
               .cmp_right_x2 = true,
               .c0_pred = true});
}

std::vector<Instruction> Assembler::Words(uint32_t base) const {
  std::vector<Instruction> words;
  words.reserve(size_);
  for (const Item& item : items_) {
    if (const auto* instruction = std::get_if<Instruction>(&item)) {
      words.push_back(*instruction);
    } else if (const auto* pass = std::get_if<Pass>(&item)) {
      Pass finished = *pass;
      if (finished.rx2 != 0) {
        finished.x2_en = true;
      }
      if (raw_copy_ && !Arithmetic(finished)) {
        finished.ialu = Ialu::kOr;
      }
      words.push_back(EncodePass(finished));
    } else {
      const auto& branch = std::get<BranchItem>(item);
      const auto found = labels_.find(branch.label);
      Require(found != labels_.end(),
              std::format("undefined label {}", branch.label));
      words.push_back(EncodeBranch(
          branch.cond, branch.ra, base + static_cast<uint32_t>(found->second)));
    }
  }
  return words;
}

}  // namespace hotarubi::compiler::vprog
