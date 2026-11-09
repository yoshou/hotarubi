#ifndef HOTARUBI_TOOLS_ISS_ENCODING_HPP_
#define HOTARUBI_TOOLS_ISS_ENCODING_HPP_

#include <cstdint>

#include "word.hpp"

namespace hotarubi::iss::encoding {

struct PassFields {
  uint32_t rx = 0;
  uint32_t rx2 = 0;
  uint32_t rd = 0;
  uint32_t rq = 0;
  uint32_t rnw = 0;
  uint32_t rn8 = 0;
  uint32_t rpb = 0;
  uint32_t rimm = 0;
  uint32_t rimm2 = 0;
  uint32_t x2_en = 0;
  uint32_t wx = 0;
  uint32_t wx2 = 0;
  uint32_t wd = 0;
  uint32_t pa = 0;
  uint32_t pa_at_mul = 0;
  uint32_t c0 = 0;
  uint32_t b = 0;
  uint32_t nl = 0;
  uint32_t c1 = 0;
  uint32_t neg_c1 = 0;
  uint32_t mask_en = 0;
  uint32_t red = 0;
  uint32_t wr = 0;
  uint32_t sc = 0;
  uint32_t px = 0;
  uint32_t pcol = 0;
  uint32_t cmp = 0;
  uint32_t cmp_src = 0;
  uint32_t cmp_int = 0;
  uint32_t sel = 0;
  uint32_t ialu = 0;
  uint32_t dst_s1 = 0;
  uint32_t c0_pred = 0;
  uint32_t nodrain = 0;
  uint32_t x_s1 = 0;
  uint32_t ctx = 0;
  uint32_t rm = 0;
};

struct SmemFields {
  uint32_t op = 0;
  uint32_t rd = 0;
  uint32_t ra = 0;
  uint32_t rb = 0;
  uint32_t col_imm = 0;
  uint32_t col = 0;
  uint32_t which = 0;
  uint32_t neg = 0;
  uint32_t ctx = 0;
  uint32_t imm = 0;
};

Word128& Set(Word128& word, uint32_t lsb, uint32_t width, uint64_t value);
Word128 Pass(const PassFields& fields);
Word128 Salu(uint32_t op, uint32_t rd, uint32_t ra, uint32_t rb);
Word128 SaluImm(uint32_t op, uint32_t rd, uint32_t ra, uint32_t imm);
Word128 Smem(const SmemFields& fields);
Word128 Branch(uint32_t cond, uint32_t ra, uint32_t target);
Word128 Ret();

}  // namespace hotarubi::iss::encoding

#endif  // HOTARUBI_TOOLS_ISS_ENCODING_HPP_
