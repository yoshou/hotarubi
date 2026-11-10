#include "vprog.hpp"

#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "isa.hpp"

namespace hotarubi::compiler::vprog {
namespace {

constexpr uint32_t kKindPass = 0;
constexpr uint32_t kKindSalu = 1;
constexpr uint32_t kKindSmem = 2;
constexpr uint32_t kKindBranch = 3;
constexpr uint32_t kKindRet = 4;
constexpr uint32_t kKindNop = 5;

void Field(Instruction& instruction, uint32_t hi, uint32_t width,
           uint32_t value) {
  instruction.Set(hi - width + 1, width, value);
}

template <typename E>
uint32_t U(E value) {
  return static_cast<uint32_t>(value);
}

uint32_t U(bool value) { return value ? 1 : 0; }

bool Arithmetic(const Pass& pass) {
  return pass.c0 != C0::kZero || pass.b != B::kOne || pass.nl != Nl::kNone ||
         pass.c1 != C1::kZero || pass.ialu != Ialu::kNone || pass.neg_c1 ||
         pass.red != Reduce::kNone || pass.pa != Pbuf::kNone;
}

}  // namespace

Instruction EncodePass(const Pass& pass) {
  Instruction instruction;
  Field(instruction, 127, 4, kKindPass);
  Field(instruction, 123, 5, pass.rx);
  Field(instruction, 118, 5, pass.rx2);
  Field(instruction, 113, 5, pass.rd);
  Field(instruction, 108, 5, pass.rq);
  Field(instruction, 103, 5, pass.rnw);
  Field(instruction, 98, 5, pass.rn8);
  Field(instruction, 93, 5, pass.rpb);
  Field(instruction, 88, 5, pass.rimm);
  Field(instruction, 83, 5, pass.rimm2);
  Field(instruction, 78, 1, U(pass.x2_en));
  Field(instruction, 77, 2, U(pass.wx));
  Field(instruction, 75, 2, U(pass.wx2));
  Field(instruction, 73, 2, U(pass.wd));
  Field(instruction, 71, 2, U(pass.pa));
  Field(instruction, 68, 3, U(pass.c0));
  Field(instruction, 65, 3, U(pass.b));
  Field(instruction, 62, 2, U(pass.nl));
  Field(instruction, 60, 2, U(pass.c1));
  Field(instruction, 58, 1, U(pass.neg_c1));
  Field(instruction, 57, 1, U(pass.mask_en));
  Field(instruction, 56, 2, U(pass.red));
  Field(instruction, 54, 2, U(pass.wr));
  Field(instruction, 52, 3, U(pass.sc));
  Field(instruction, 49, 3, U(pass.px));
  Field(instruction, 43, 3, pass.pcol);
  Field(instruction, 40, 3, U(pass.cmp));
  Field(instruction, 37, 2, U(pass.cmp_src));
  Field(instruction, 35, 1, U(pass.cmp_int));
  Field(instruction, 34, 3, U(pass.sel));
  Field(instruction, 31, 3, U(pass.ialu));
  Field(instruction, 28, 1, U(pass.dst_s1));
  Field(instruction, 27, 1, U(pass.c0_pred));
  Field(instruction, 25, 1, U(pass.x_s1));
  Field(instruction, 24, 2, pass.ctx);
  Field(instruction, 22, 2, U(pass.rm));
  return instruction;
}

Instruction EncodeSalu(Salu op, uint32_t rd, uint32_t ra, uint32_t rb,
                       std::optional<uint32_t> imm) {
  Instruction instruction;
  Field(instruction, 127, 4, kKindSalu);
  Field(instruction, 123, 5, U(op));
  Field(instruction, 118, 5, rd);
  Field(instruction, 113, 5, ra);
  Field(instruction, 108, 5, rb);
  if (imm.has_value()) {
    Field(instruction, 103, 1, 1);
    Field(instruction, 31, 32, *imm);
  }
  return instruction;
}

Instruction EncodeSmem(Smem op, const SmemArgs& args) {
  Instruction instruction;
  Field(instruction, 127, 4, kKindSmem);
  Field(instruction, 123, 3, U(op));
  Field(instruction, 120, 5, args.rd);
  Field(instruction, 115, 5, args.ra);
  Field(instruction, 110, 5, args.rb);
  if (args.col.has_value()) {
    Field(instruction, 105, 1, 1);
    Field(instruction, 104, 5, *args.col);
  }
  Field(instruction, 99, 1, U(args.which));
  Field(instruction, 97, 2, args.ctx);
  Field(instruction, 31, 32, args.imm);
  return instruction;
}

Instruction EncodeBranch(Cond cond, uint32_t ra, uint32_t target) {
  Instruction instruction;
  Field(instruction, 127, 4, kKindBranch);
  Field(instruction, 123, 2, U(cond));
  Field(instruction, 121, 5, ra);
  Field(instruction, 15, 16, target);
  return instruction;
}

Instruction EncodeRet() {
  Instruction instruction;
  Field(instruction, 127, 4, kKindRet);
  return instruction;
}

Instruction EncodeNop() {
  Instruction instruction;
  Field(instruction, 127, 4, kKindNop);
  return instruction;
}

void Assembler::Label(std::string_view name) {
  Require(!labels_.contains(name), std::format("duplicate label {}", name));
  labels_.emplace(std::string(name), size_);
}

Assembler& Assembler::Emit(const Pass& pass) {
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
                                 uint32_t dst_word_stride, TransposePack pack) {
  Instruction word;
  word.Set(124, 4, 7)
      .Set(119, 5, src)
      .Set(114, 5, src_word_stride)
      .Set(109, 5, dst)
      .Set(104, 5, tiles)
      .Set(99, 5, dst_tile_stride)
      .Set(94, 5, src_tile_stride)
      .Set(89, 5, dst_word_stride)
      .Set(88, 1, static_cast<uint32_t>(pack));
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
               .cmp_src = CmpSrc::kX2Imm2,
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
               .cmp_src = CmpSrc::kX2,
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
