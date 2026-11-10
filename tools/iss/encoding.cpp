#include "encoding.hpp"

#include <cstdint>
#include <format>

#include "hotarubi/require.hpp"
#include "word.hpp"

namespace hotarubi::iss::encoding {
namespace {

constexpr uint32_t kKindPass = 0;
constexpr uint32_t kKindSalu = 1;
constexpr uint32_t kKindSmem = 2;
constexpr uint32_t kKindBranch = 3;
constexpr uint32_t kKindRet = 4;

Word128 Kind(uint32_t kind) {
  Word128 word;
  return Set(word, 124, 4, kind);
}

}  // namespace

Word128& Set(Word128& word, uint32_t lsb, uint32_t width, uint64_t value) {
  Require(width == 64 || value < (uint64_t{1} << width),
          std::format("{:#x} does not fit in {} bits", value, width));
  for (uint32_t i = 0; i < width; ++i) {
    const uint32_t bit = lsb + i;
    uint64_t& part = bit < 64 ? word.low : word.high;
    const uint64_t mask = uint64_t{1} << (bit % 64);
    part = ((value >> i) & 1U) != 0 ? (part | mask) : (part & ~mask);
  }
  return word;
}

Word128 Pass(const PassFields& f) {
  Word128 w = Kind(kKindPass);
  Set(w, 119, 5, f.rx);
  Set(w, 114, 5, f.rx2);
  Set(w, 109, 5, f.rd);
  Set(w, 104, 5, f.rq);
  Set(w, 99, 5, f.rnw);
  Set(w, 94, 5, f.rn8);
  Set(w, 89, 5, f.rpb);
  Set(w, 84, 5, f.rimm);
  Set(w, 79, 5, f.rimm2);
  Set(w, 78, 1, f.x2_en);
  Set(w, 76, 2, f.mg);
  Set(w, 74, 2, f.ms);
  Set(w, 72, 2, f.mx);
  Set(w, 70, 2, f.mx2);
  Set(w, 68, 2, f.md);
  Set(w, 66, 2, f.pa);
  Set(w, 63, 3, f.c0);
  Set(w, 60, 3, f.b);
  Set(w, 58, 2, f.nl);
  Set(w, 56, 2, f.po);
  Set(w, 55, 1, f.pv);
  Set(w, 54, 1, f.mask_en);
  Set(w, 52, 2, f.red);
  Set(w, 50, 2, f.wr);
  Set(w, 47, 3, f.sf);
  Set(w, 46, 1, f.st);
  Set(w, 45, 1, f.srn);
  Set(w, 44, 1, f.seps);
  Set(w, 24, 20, f.px);
  Set(w, 21, 3, f.cmp);
  Set(w, 19, 2, f.cl);
  Set(w, 18, 1, f.cr);
  Set(w, 17, 1, f.cmp_int);
  Set(w, 14, 3, f.sel);
  Set(w, 11, 3, f.ialu);
  Set(w, 10, 1, f.c0_pred);
  Set(w, 9, 1, f.x_s1);
  Set(w, 8, 1, f.dst_s1);
  Set(w, 6, 2, f.ctx);
  return Set(w, 4, 2, f.rm);
}

Word128 Salu(uint32_t op, uint32_t rd, uint32_t ra, uint32_t rb) {
  Word128 w = Kind(kKindSalu);
  Set(w, 119, 5, op);
  Set(w, 114, 5, rd);
  Set(w, 109, 5, ra);
  return Set(w, 104, 5, rb);
}

Word128 SaluImm(uint32_t op, uint32_t rd, uint32_t ra, uint32_t imm) {
  Word128 w = Kind(kKindSalu);
  Set(w, 119, 5, op);
  Set(w, 114, 5, rd);
  Set(w, 109, 5, ra);
  Set(w, 103, 1, 1);
  return Set(w, 0, 32, imm);
}

Word128 Smem(const SmemFields& f) {
  Word128 w = Kind(kKindSmem);
  Set(w, 121, 3, f.op);
  Set(w, 116, 5, f.rd);
  Set(w, 111, 5, f.ra);
  Set(w, 106, 5, f.rb);
  Set(w, 105, 1, f.col_imm);
  Set(w, 100, 5, f.col);
  Set(w, 99, 1, f.which);
  Set(w, 96, 2, f.ctx);
  return Set(w, 0, 32, f.imm);
}

Word128 Branch(uint32_t cond, uint32_t ra, uint32_t target) {
  Word128 w = Kind(kKindBranch);
  Set(w, 122, 2, cond);
  Set(w, 117, 5, ra);
  return Set(w, 0, 16, target);
}

Word128 Ret() { return Kind(kKindRet); }

}  // namespace hotarubi::iss::encoding
