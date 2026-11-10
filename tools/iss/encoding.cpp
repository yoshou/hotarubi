#include "encoding.hpp"

#include <cstdint>
#include <format>

#include "hotarubi/isa.hpp"
#include "hotarubi/require.hpp"
#include "word.hpp"

namespace hotarubi::iss::encoding {
namespace {

namespace pg = isa::program;

Word128 Kind(uint32_t kind) {
  Word128 word;
  return Set(word, pg::kKind, kind);
}

}  // namespace

Word128& Set(Word128& word, isa::Field field, uint64_t value) {
  Require(field.width == 64 || value < (uint64_t{1} << field.width),
          std::format("{:#x} does not fit in {} bits", value, field.width));
  for (uint32_t i = 0; i < field.width; ++i) {
    const uint32_t bit = field.lsb + i;
    uint64_t& part = bit < 64 ? word.low : word.high;
    const uint64_t mask = uint64_t{1} << (bit % 64);
    part = ((value >> i) & 1U) != 0 ? (part | mask) : (part & ~mask);
  }
  return word;
}

Word128 Pass(const PassFields& f) {
  namespace p = isa::pass;
  Word128 w = Kind(pg::kKindPass);
  Set(w, p::kRx, f.rx);
  Set(w, p::kRx2, f.rx2);
  Set(w, p::kRd, f.rd);
  Set(w, p::kRq, f.rq);
  Set(w, p::kRnw, f.rnw);
  Set(w, p::kRn8, f.rn8);
  Set(w, p::kRpb, f.rpb);
  Set(w, p::kRimm, f.rimm);
  Set(w, p::kRimm2, f.rimm2);
  Set(w, p::kX2En, f.x2_en);
  Set(w, p::kMg, f.mg);
  Set(w, p::kMs, f.ms);
  Set(w, p::kMx, f.mx);
  Set(w, p::kMx2, f.mx2);
  Set(w, p::kMd, f.md);
  Set(w, p::kPa, f.pa);
  Set(w, p::kC0, f.c0);
  Set(w, p::kB, f.b);
  Set(w, p::kNl, f.nl);
  Set(w, p::kPo, f.po);
  Set(w, p::kPv, f.pv);
  Set(w, p::kRed, f.red);
  Set(w, p::kWr, f.wr);
  Set(w, p::kSf, f.sf);
  Set(w, p::kSt, f.st);
  Set(w, p::kSrn, f.srn);
  Set(w, p::kSeps, f.seps);
  Set(w, p::kPx, f.px);
  Set(w, p::kCmp, f.cmp);
  Set(w, p::kCl, f.cl);
  Set(w, p::kCr, f.cr);
  Set(w, p::kCmpInt, f.cmp_int);
  Set(w, p::kSel, f.sel);
  Set(w, p::kIalu, f.ialu);
  Set(w, p::kC0Pred, f.c0_pred);
  Set(w, p::kXS1, f.x_s1);
  Set(w, p::kDstS1, f.dst_s1);
  Set(w, p::kCtx, f.ctx);
  return Set(w, p::kRm, f.rm);
}

Word128 Salu(uint32_t op, uint32_t rd, uint32_t ra, uint32_t rb) {
  Word128 w = Kind(pg::kKindSalu);
  Set(w, pg::kOp, op);
  Set(w, pg::kRd, rd);
  Set(w, pg::kRa, ra);
  return Set(w, pg::kRb, rb);
}

Word128 SaluImm(uint32_t op, uint32_t rd, uint32_t ra, uint32_t imm) {
  Word128 w = Kind(pg::kKindSalu);
  Set(w, pg::kOp, op);
  Set(w, pg::kRd, rd);
  Set(w, pg::kRa, ra);
  Set(w, pg::kUseImm, 1);
  return Set(w, pg::kImm, imm);
}

Word128 Smem(const SmemFields& f) {
  Word128 w = Kind(pg::kKindSmem);
  Set(w, pg::kOp, f.op);
  Set(w, pg::kRd, f.rd);
  Set(w, pg::kRa, f.ra);
  Set(w, pg::kRb, f.rb);
  Set(w, pg::kUseImm, f.col_imm);
  Set(w, pg::kSmemCol, f.col);
  Set(w, pg::kSmemWhich, f.which);
  Set(w, pg::kSmemCtx, f.ctx);
  return Set(w, pg::kImm, f.imm);
}

Word128 Branch(uint32_t cond, uint32_t ra, uint32_t target) {
  Word128 w = Kind(pg::kKindBranch);
  Set(w, pg::kOp, cond);
  Set(w, pg::kRa, ra);
  return Set(w, pg::kImm, target);
}

Word128 Ret() { return Kind(pg::kKindRet); }

}  // namespace hotarubi::iss::encoding
