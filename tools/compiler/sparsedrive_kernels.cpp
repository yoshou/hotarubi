#include "sparsedrive_kernels.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hotarubi/exact.hpp"
#include "isa.hpp"
#include "tensors.hpp"
#include "vprog.hpp"

namespace hotarubi::compiler::sparsedrive {
namespace {

namespace vp = vprog;
using vp::Pass;

constexpr uint32_t kMagic = 0x7EF311C7U;
constexpr uint32_t kSignBit = 0x80000000U;
constexpr uint32_t kI2f0 = 0x4B400000U;
constexpr int64_t kCf = 12582912;
constexpr uint32_t kNullEntry = 0xFFFFFFFFU;
constexpr uint32_t kFsMaxMicroseconds = 2000000;
constexpr uint32_t kDetGroups = 6;
constexpr uint32_t kMapStride = 8;
constexpr uint32_t kMapWorkGroups = 6;

constexpr uint32_t kIkBaseX = 0;
constexpr uint32_t kIkTbX = 1;
constexpr uint32_t kIkBaseA = 2;
constexpr uint32_t kIkTbA = 3;
constexpr uint32_t kIkFna = 4;
constexpr uint32_t kIkNinf = 5;
constexpr uint32_t kIkOne = 6;
constexpr uint32_t kIkNone = 7;
constexpr uint32_t kIkI2f0 = 8;
constexpr uint32_t kIkCf = 9;
constexpr uint32_t kIkZero = 10;
constexpr uint32_t kIkFntemp = 11;
constexpr uint32_t kIkNhalf = 12;
constexpr uint32_t kIkSign = 13;
constexpr uint32_t kIkHalf = 14;
constexpr uint32_t kIkDecay = 15;

constexpr uint32_t kQkLthr = 0;
constexpr uint32_t kQkFna = 1;
constexpr uint32_t kQkOne = 2;
constexpr uint32_t kQkNeg = 3;
constexpr uint32_t kQkFtq = 4;
constexpr uint32_t kQlTb4 = 0;
constexpr uint32_t kQlBaseO = 1;
constexpr uint32_t kQlTb1 = 2;
constexpr uint32_t kQlTbC = 5;
constexpr uint32_t kQlBaseC = 6;
constexpr uint32_t kQlI2f0 = 7;
constexpr uint32_t kQlCf = 8;
constexpr uint32_t kQlFna = 9;
constexpr uint32_t kQsFnt = 0;
constexpr uint32_t kQsTb = 1;
constexpr uint32_t kQsBase = 2;
constexpr uint32_t kQsENa = 3;
constexpr uint32_t kQsI2f0 = 4;
constexpr uint32_t kQsCf = 5;

uint32_t Decimal(int64_t numerator, int64_t denominator) {
  return exact::Real::Binary64(
             exact::Real::Ratio(numerator, denominator).ToBinary64())
      .ToBinary32();
}

uint32_t NegativeHuge() {
  constexpr int64_t kHalfDigits = 1000000000000000;
  return exact::Real::Binary64((exact::Real::Integer(kHalfDigits) *
                                exact::Real::Integer(kHalfDigits))
                                   .Negated()
                                   .ToBinary64())
      .ToBinary32();
}

struct Plane {
  std::string_view name;
  uint32_t g = 0;
  uint32_t k = 0;
};

constexpr std::array<Plane, 30> kDetPlanes = {{
    {.name = "SX", .g = 0, .k = 0},  {.name = "SY", .g = 0, .k = 1},
    {.name = "SZ", .g = 0, .k = 2},  {.name = "ONES", .g = 0, .k = 3},
    {.name = "DHP", .g = 1, .k = 0}, {.name = "DWP", .g = 1, .k = 1},
    {.name = "LQX", .g = 1, .k = 2}, {.name = "LQY", .g = 1, .k = 3},
    {.name = "LQZ", .g = 2, .k = 0}, {.name = "OX", .g = 2, .k = 1},
    {.name = "OY", .g = 2, .k = 2},  {.name = "OZ", .g = 2, .k = 3},
    {.name = "T1", .g = 3, .k = 0},  {.name = "T2", .g = 3, .k = 1},
    {.name = "KX", .g = 3, .k = 2},  {.name = "KY", .g = 3, .k = 3},
    {.name = "KZ", .g = 4, .k = 0},  {.name = "VM", .g = 4, .k = 1},
    {.name = "P3", .g = 4, .k = 2},  {.name = "IM", .g = 4, .k = 3},
    {.name = "T3", .g = 5, .k = 0},  {.name = "U", .g = 5, .k = 1},
    {.name = "V", .g = 5, .k = 2},   {.name = "Y", .g = 5, .k = 3},
    {.name = "HH", .g = 3, .k = 2},  {.name = "WW", .g = 3, .k = 3},
    {.name = "P1", .g = 4, .k = 0},  {.name = "P2", .g = 1, .k = 2},
    {.name = "LH", .g = 1, .k = 3},  {.name = "LW", .g = 2, .k = 0},
}};

constexpr std::array<Plane, 48> kMapPlanes = {{
    {.name = "ONES", .g = 0, .k = 0},  {.name = "ANCX0", .g = 0, .k = 1},
    {.name = "ANCX1", .g = 0, .k = 2}, {.name = "ANCX2", .g = 0, .k = 3},
    {.name = "ANCY0", .g = 0, .k = 4}, {.name = "ANCY1", .g = 0, .k = 5},
    {.name = "ANCY2", .g = 0, .k = 6}, {.name = "T3", .g = 0, .k = 7},
    {.name = "KX", .g = 1, .k = 0},    {.name = "KY", .g = 1, .k = 1},
    {.name = "KZ", .g = 1, .k = 2},    {.name = "U", .g = 1, .k = 3},
    {.name = "V", .g = 1, .k = 4},     {.name = "W", .g = 1, .k = 5},
    {.name = "Y", .g = 1, .k = 6},     {.name = "VM", .g = 1, .k = 7},
    {.name = "T1", .g = 2, .k = 0},    {.name = "T2", .g = 2, .k = 1},
    {.name = "HH", .g = 2, .k = 2},    {.name = "WW", .g = 2, .k = 3},
    {.name = "P1", .g = 2, .k = 4},    {.name = "P2", .g = 2, .k = 5},
    {.name = "LH", .g = 2, .k = 6},    {.name = "LW", .g = 2, .k = 7},
    {.name = "LH1", .g = 3, .k = 0},   {.name = "LW1", .g = 3, .k = 1},
    {.name = "HI", .g = 3, .k = 2},    {.name = "WI", .g = 3, .k = 3},
    {.name = "IM", .g = 3, .k = 4},    {.name = "BWK", .g = 3, .k = 5},
    {.name = "T4", .g = 3, .k = 6},    {.name = "T5", .g = 3, .k = 7},
    {.name = "A0", .g = 4, .k = 0},    {.name = "A1", .g = 4, .k = 1},
    {.name = "A2", .g = 4, .k = 2},    {.name = "A3", .g = 4, .k = 3},
    {.name = "B0", .g = 4, .k = 4},    {.name = "B1", .g = 4, .k = 5},
    {.name = "B2", .g = 4, .k = 6},    {.name = "B3", .g = 4, .k = 7},
    {.name = "O0", .g = 5, .k = 0},    {.name = "O1", .g = 5, .k = 1},
    {.name = "O2", .g = 5, .k = 2},    {.name = "O3", .g = 5, .k = 3},
    {.name = "O4", .g = 5, .k = 4},    {.name = "O5", .g = 5, .k = 5},
    {.name = "O6", .g = 5, .k = 6},    {.name = "O7", .g = 5, .k = 7},
}};

template <size_t N>
const Plane& FindPlane(const std::array<Plane, N>& planes,
                       std::string_view name) {
  const auto* found = std::ranges::find(planes, name, &Plane::name);
  Require(found != planes.end(), std::format("no work plane {}", name));
  return *found;
}

class Keypoint {
 public:
  Keypoint(vp::Assembler& a, uint32_t tq, bool map)
      : a_(a), tq_(tq), map_(map) {}

  void Prologue() {
    a_.Move(18, 5);
    a_.LoadImmediate(5, map_ ? kMapStride : 4);
    a_.LoadImmediate(28, 1);
    a_.Alu(vp::Salu::kShr, 10, 8, 0, 12);
    a_.Alu(vp::Salu::kAnd, 10, 10, 0, 0xFFF);
    if (map_) {
      a_.Alu(vp::Salu::kShr, 20, 8, 0, 24);
      a_.Alu(vp::Salu::kAnd, 20, 20, 0, 7);
      a_.Alu(vp::Salu::kShr, 21, 8, 0, 27);
      a_.Alu(vp::Salu::kAnd, 21, 21, 0, 3);
      a_.Alu(vp::Salu::kShr, 22, 8, 0, 29);
      a_.Alu(vp::Salu::kAnd, 22, 22, 0, 3);
      a_.Alu(vp::Salu::kAnd, 7, 8, 0, 0xFFF);
      a_.Alu(vp::Salu::kAdd, 9, 7, 0, 1);
      a_.Alu(vp::Salu::kAdd, 19, 7, 0, 2);
    } else {
      a_.Alu(vp::Salu::kAnd, 7, 8, 0, 0xFFF);
      a_.Alu(vp::Salu::kAdd, 8, 7, 0, 1);
      a_.Alu(vp::Salu::kAdd, 9, 7, 0, 2);
    }
  }

  uint32_t Pl(uint32_t reg, std::string_view name) {
    const Plane& plane =
        map_ ? FindPlane(kMapPlanes, name) : FindPlane(kDetPlanes, name);
    a_.Alu(vp::Salu::kAdd, reg, 10, 0,
           (plane.g * (map_ ? kMapStride : 4) * tq_) + plane.k);
    return reg;
  }

  [[nodiscard]] uint32_t PlaneOffset(std::string_view name) const {
    const Plane& plane = FindPlane(kMapPlanes, name);
    return (plane.g * kMapStride * tq_) + plane.k;
  }

  void P(Pass pass) {
    pass.rq = 4;
    pass.rnw = 28;
    pass.rn8 = 5;
    pass.wr = vp::Write::kFbuf;
    a_.Emit(pass);
  }

  void Ld(uint32_t rd, uint32_t column) {
    a_.Memory(vp::Smem::kSld, {.rd = rd, .ra = 7, .col = column});
  }

  void Project(uint32_t x, uint32_t y, uint32_t z,
               const std::array<std::string_view, 3>& outputs) {
    for (uint32_t r = 0; r < 3; ++r) {
      Pl(11, outputs[r]);
      Ld(14, kConstProjection + (4 * r));
      P({.rx = x, .rd = 11, .rimm = 14, .b = vp::B::kImm});
      Ld(14, kConstProjection + (4 * r) + 1);
      P({.rx = y,
         .rx2 = 11,
         .rd = 11,
         .rimm = 14,
         .b = vp::B::kImm,
         .c1 = vp::C1::kX2});
      Ld(14, kConstProjection + (4 * r) + 2);
      P({.rx = z,
         .rx2 = 11,
         .rd = 11,
         .rimm = 14,
         .b = vp::B::kImm,
         .c1 = vp::C1::kX2});
      Ld(14, kConstProjection + (4 * r) + 3);
      P({.rx = 11, .rd = 11, .rimm = 14, .c0 = vp::C0::kImm});
    }
  }

  void Location(std::string_view w) {
    Pl(11, w);
    Pl(12, "Y");
    Pl(13, "T2");
    Ld(14, kConstEps);
    P({.rx = 11,
       .rd = 11,
       .rimm = 14,
       .rimm2 = 14,
       .cmp = vp::Cmp::kGe,
       .cmp_src = vp::CmpSrc::kImm2,
       .sel = vp::Sel::kImm});
    Ld(14, kConstMagic);
    P({.rx = 11,
       .rd = 12,
       .rimm = 14,
       .c0 = vp::C0::kImm,
       .ialu = vp::Ialu::kRsub});
    Ld(14, kConstNtwo);
    Ld(15, kConstNone);
    for (int i = 0; i < 4; ++i) {
      P({.rx = 11, .rx2 = 12, .rd = 13, .b = vp::B::kX2});
      P({.rx = 13, .rd = 13, .rimm = 14, .c0 = vp::C0::kImm});
      P({.rx = 13, .rd = 13, .rimm = 15, .b = vp::B::kImm});
      P({.rx = 12, .rx2 = 13, .rd = 12, .b = vp::B::kX2});
    }
    Pl(21, "U");
    Pl(22, "V");
    P({.rx = 21, .rx2 = 12, .rd = 21, .b = vp::B::kX2});
    P({.rx = 22, .rx2 = 12, .rd = 22, .b = vp::B::kX2});
    Ld(14, kConstRwx);
    P({.rx = 21, .rd = 21, .rimm = 14, .b = vp::B::kImm});
    Ld(14, kConstRwy);
    P({.rx = 22, .rd = 22, .rimm = 14, .b = vp::B::kImm});
    Pl(11, "ONES");
    Pl(12, "VM");
    Ld(14, kConstZero);
    Ld(15, kConstOne);
    Inside(11, 12, 21, 14, vp::Cmp::kGt);
    Inside(12, 12, 21, 15, vp::Cmp::kLt);
    Inside(12, 12, 22, 14, vp::Cmp::kGt);
    Inside(12, 12, 22, 15, vp::Cmp::kLt);
  }

  void Inside(uint32_t rx, uint32_t rd, uint32_t rx2, uint32_t bound,
              vp::Cmp cmp) {
    P({.rx = rx,
       .rx2 = rx2,
       .rd = rd,
       .rimm2 = bound,
       .cmp = cmp,
       .cmp_src = vp::CmpSrc::kX2Imm2,
       .sel = vp::Sel::kZero});
  }

  void Floors(uint32_t r_none) {
    a_.FloorFloat(11, 23, 4, 28, 5, 16, 17, r_none, 25);
    a_.FloorFloat(12, 24, 4, 28, 5, 16, 17, r_none, 25);
  }

  void HalfPixel() {
    Pl(21, "U");
    Pl(22, "V");
    Pl(11, "HH");
    Pl(12, "WW");
    Ld(16, kConstNhalf);
    P({.rx = 22, .rd = 11, .rimm = 14, .b = vp::B::kImm});
    P({.rx = 11, .rd = 11, .rimm = 16, .c0 = vp::C0::kImm});
    P({.rx = 21, .rd = 12, .rimm = 15, .b = vp::B::kImm});
    P({.rx = 12, .rd = 12, .rimm = 16, .c0 = vp::C0::kImm});
  }

 private:
  vp::Assembler& a_;
  uint32_t tq_;
  bool map_;
};

}  // namespace

uint32_t FloatBits(int64_t value) {
  return exact::Real::Integer(value).ToBinary32();
}

Word ConstWord() {
  Word w{};
  w[kConstOne] = FloatBits(1);
  w[kConstNone] = FloatBits(-1);
  w[kConstNhalf] = Decimal(-1, 2);
  w[kConstTwo] = FloatBits(2);
  w[kConstNtwo] = FloatBits(-2);
  w[kConstEps] = Decimal(1, 100000);
  w[kConstCf] = FloatBits(kCf);
  w[kConstNcf] = FloatBits(-kCf);
  w[kConstHalf] = Decimal(1, 2);
  w[kConstRwx] = FloatBits(1);
  w[kConstRwy] = FloatBits(1);
  w[kConstZero] = 0;
  w[kConstMagic] = kMagic;
  w[kConstSign] = kSignBit;
  w[kConstI2f0] = kI2f0;
  return w;
}

Word FixScaleWord(const std::vector<uint32_t>& fix_scale) {
  Require(fix_scale.size() == 21, "fix_scale is not 7 x 3");
  Word w{};
  for (uint32_t p = 0; p < 7; ++p) {
    for (uint32_t c = 0; c < 3; ++c) {
      w[(8 * c) + p] = fix_scale[(3 * p) + c];
    }
  }
  return w;
}

Word LevelWord(const std::vector<Level>& levels) {
  Word w{};
  for (size_t l = 0; l < levels.size(); ++l) {
    w[4 * l] = FloatBits(levels[l].h);
    w[(4 * l) + 1] = FloatBits(levels[l].w);
    w[(4 * l) + 2] = static_cast<uint32_t>(levels[l].phys / 16);
    w[(4 * l) + 3] = FloatBits(static_cast<int64_t>(levels[l].tb / 16));
  }
  return w;
}

Word ZhWord() {
  const exact::Real ground = exact::Real::Binary32(Decimal(-184023, 100000));
  constexpr std::array<std::pair<int64_t, int64_t>, kMapHeights> kHeights = {
      {{0, 1}, {1, 2}, {-1, 2}, {1, 1}, {-1, 1}}};
  Word w{};
  for (uint32_t h = 0; h < kMapHeights; ++h) {
    w[h] = (ground + exact::Real::Ratio(kHeights[h].first, kHeights[h].second))
               .ToBinary32();
  }
  return w;
}

std::vector<Word> LaneConstantWords() {
  std::vector<Word> words(4, Word{});
  for (uint32_t lane = 0; lane < 8; ++lane) {
    words[0][lane] = FloatBits(lane >= 4 ? 1 : 0);
    words[1][lane] = FloatBits((lane / 2) % 2);
    words[2][lane] = lane % 2 == 0 ? kNullEntry : 0;
    words[3][lane] = lane % 2 == 0 ? 0 : kNullEntry;
  }
  return words;
}

Word IbankConstWord(uint32_t base_x16, uint32_t tb_x16, uint32_t base_a16,
                    uint32_t tb_a16, uint32_t na, uint32_t ntemp) {
  Word w{};
  w[kIkBaseX] = base_x16;
  w[kIkTbX] = FloatBits(tb_x16);
  w[kIkBaseA] = base_a16;
  w[kIkTbA] = FloatBits(tb_a16);
  w[kIkFna] = FloatBits(na);
  w[kIkFntemp] = FloatBits(ntemp);
  w[kIkNinf] = 0xFF800000U;
  w[kIkOne] = FloatBits(1);
  w[kIkNone] = FloatBits(-1);
  w[kIkI2f0] = kI2f0;
  w[kIkCf] = FloatBits(kCf);
  w[kIkZero] = 0;
  w[kIkNhalf] = Decimal(-1, 2);
  w[kIkSign] = kSignBit;
  w[kIkHalf] = Decimal(1, 2);
  w[kIkDecay] = Decimal(6, 10);
  return w;
}

std::vector<Word> LaneIndexWords(uint32_t n) {
  std::vector<Word> words(n, Word{});
  for (uint32_t i = 0; i < n * 32; ++i) {
    words[i / 32][i % 32] = FloatBits(i);
  }
  return words;
}

Word LaneIdWord() {
  Word w{};
  for (uint32_t i = 0; i < 32; ++i) {
    w[i] = i;
  }
  return w;
}

Word FsCol8Word() {
  Word w{};
  for (uint32_t i = 0; i < 32; ++i) {
    w[i] = i % 8;
  }
  return w;
}

std::vector<Word> FsDetTemplate() {
  std::vector<Word> words(18, Word{});
  for (uint32_t i = 3; i < 6; ++i) {
    words[i][i] = FloatBits(1);
  }
  return words;
}

Word IqueueConstWord(uint32_t na, uint32_t tq) {
  const exact::Real threshold =
      exact::Real::Binary64(exact::Real::Ratio(1, 5).ToBinary64());
  const exact::Real rest =
      exact::Real::Binary64((exact::Real::Integer(1) - threshold).ToBinary64());
  const exact::Real odds =
      exact::Real::Binary64((threshold / rest).ToBinary64());
  Word w{};
  w[kQkLthr] =
      exact::Real::Binary64(exact::Log(odds).ToBinary64()).ToBinary32();
  w[kQkFna] = FloatBits(na);
  w[kQkOne] = FloatBits(1);
  w[kQkNeg] = NegativeHuge();
  w[kQkFtq] = FloatBits(tq);
  return w;
}

Word IqueueListWord(uint32_t base_old16, uint32_t tb16, uint32_t base_cur16,
                    uint32_t tb_cur16, uint32_t na) {
  Word w{};
  w[kQlTb4] = FloatBits(int64_t{4} * tb16);
  w[kQlBaseO] = base_old16;
  for (uint32_t k = 1; k <= 3; ++k) {
    w[kQlTb1 + k - 1] = k * tb16;
  }
  w[kQlTbC] = FloatBits(tb_cur16);
  w[kQlBaseC] = base_cur16;
  w[kQlI2f0] = kI2f0;
  w[kQlCf] = FloatBits(kCf);
  w[kQlFna] = FloatBits(na);
  return w;
}

Word IqueueStlistWord(uint32_t base16, uint32_t tb16, uint32_t nt,
                      uint32_t na) {
  Word w{};
  w[kQsFnt] = FloatBits(nt);
  w[kQsTb] = FloatBits(tb16);
  w[kQsBase] = base16;
  w[kQsENa] = base16 + (na * tb16);
  w[kQsI2f0] = kI2f0;
  w[kQsCf] = FloatBits(kCf);
  return w;
}

Word IqueueTWord() {
  Word w{};
  for (uint32_t i = 0; i < 8; ++i) {
    w[i] = FloatBits(i % 4);
  }
  return w;
}

std::vector<uint64_t> SinTbuf() {
  std::vector<uint64_t> table = MakeTbuf(Activation::kSilu);
  const exact::Real two_pi =
      exact::Real::Binary64(exact::Pi().ToBinary64()).Scaled(1);
  std::vector<uint32_t> v(kTableSegments + 1);
  for (uint32_t i = 0; i <= kTableSegments; ++i) {
    const exact::Real x = exact::Real::Binary64(
        (two_pi * exact::Real::Ratio(i, kTableSegments)).ToBinary64());
    v[i] = exact::Real::Binary64(exact::Sin(x).ToBinary64()).ToBinary32();
  }
  for (uint32_t i = 0; i < kTableSegments; ++i) {
    const uint32_t slope =
        (exact::Real::Binary32(v[i + 1]) - exact::Real::Binary32(v[i]))
            .ToBinary32();
    table[i] = (static_cast<uint64_t>(slope) << 32) | v[i];
  }
  return table;
}

std::vector<uint32_t> EmbedPbuf() {
  constexpr uint32_t kHalf = 128;
  std::vector<uint32_t> out(size_t{32} * 32);
  for (uint32_t i = 0; i < kHalf; ++i) {
    const exact::Real dim =
        exact::Real::Binary64(exact::Pow(exact::Real::Integer(10000),
                                         exact::Real::Ratio(i / 2, kHalf / 2))
                                  .ToBinary64());
    const uint32_t inv =
        exact::Real::Binary64((exact::Real::Integer(1) / dim).ToBinary64())
            .ToBinary32();
    const uint32_t g = i / 8;
    out[(g * 32) + (i % 8)] = inv;
    out[((16 + g) * 32) + (i % 8)] = i % 2 == 0 ? 0 : Decimal(1, 4);
  }
  return out;
}

std::vector<int64_t> LfcPerm(uint32_t learn) {
  std::vector<int64_t> perm(static_cast<size_t>(3) * learn);
  for (uint32_t q = 0; q < learn; ++q) {
    for (uint32_t c = 0; c < 3; ++c) {
      perm[(3 * q) + c] = (8 * c) + q;
    }
  }
  return perm;
}

std::vector<int64_t> MapLfcPerm() {
  std::vector<int64_t> perm(static_cast<size_t>(kMapSamples) * kMapHeights *
                            kMapLearn * 2);
  for (uint32_t s = 0; s < kMapSamples; ++s) {
    for (uint32_t h = 0; h < kMapHeights; ++h) {
      for (uint32_t lq = 0; lq < kMapLearn; ++lq) {
        for (uint32_t c = 0; c < 2; ++c) {
          perm[((((s * kMapHeights) + h) * kMapLearn + lq) * 2) + c] =
              (((h * kMapLearn) + lq) * kMapBlock) +
              (((c * kMapGroupsPerBlock) + (s / 8)) * 8) + (s % 8);
        }
      }
    }
  }
  return perm;
}

std::array<uint32_t, 3> MapGroup(uint32_t gi) {
  return {gi / (kMapLearn * kMapGroupsPerBlock),
          (gi / kMapGroupsPerBlock) % kMapLearn, gi % kMapGroupsPerBlock};
}

uint32_t DetKeypointWork(uint32_t tq) { return kDetGroups * 4 * tq; }

uint32_t MapKeypointWork(uint32_t tq) {
  return kMapWorkGroups * kMapStride * tq;
}

vp::Assembler DetKeypointInit(uint32_t tq) {
  vp::Assembler a(true);
  Keypoint k(a, tq, false);
  k.Prologue();
  k.Pl(11, "ONES");
  k.Pl(12, "DHP");
  k.Pl(13, "DWP");
  k.Ld(14, kConstOne);
  k.P({.rx = 1,
       .rd = 11,
       .rimm = 0,
       .c0 = vp::C0::kImm,
       .ialu = vp::Ialu::kAnd});
  k.P({.rx = 11, .rd = 11, .rimm = 14, .c0 = vp::C0::kImm});
  k.P({.rx = 11,
       .rd = 12,
       .rimm = 0,
       .c0 = vp::C0::kImm,
       .ialu = vp::Ialu::kAnd});
  k.P({.rx = 12,
       .rd = 12,
       .rpb = 6,
       .pa = vp::Pbuf::kRowVec,
       .c0 = vp::C0::kPbuf});
  a.Alu(vp::Salu::kAdd, 29, 6, 0, 1);
  k.P({.rx = 11,
       .rd = 13,
       .rimm = 0,
       .c0 = vp::C0::kImm,
       .ialu = vp::Ialu::kAnd});
  k.P({.rx = 13,
       .rd = 13,
       .rpb = 29,
       .pa = vp::Pbuf::kRowVec,
       .c0 = vp::C0::kPbuf});
  for (const auto& [name, col] :
       {std::pair{"SX", 3U}, {"SY", 4U}, {"SZ", 5U}}) {
    k.Pl(11, name);
    k.P({.rx = 1,
         .rd = 11,
         .b = vp::B::kLog2e,
         .nl = vp::Nl::kExp2,
         .px = vp::Perm::kBcast,
         .pcol = col});
  }
  a.Ret();
  return a;
}

vp::Assembler DetKeypointPoint(uint32_t tq, uint32_t learn) {
  vp::Assembler a(true);
  Keypoint k(a, tq, false);
  k.Prologue();
  a.Alu(vp::Salu::kSlt, 29, 18, 0, 7);
  a.Branch(vp::Cond::kZ, 29, "learn");
  constexpr std::array<std::pair<std::string_view, std::string_view>, 3>
      kFixed = {{{"SX", "OX"}, {"SY", "OY"}, {"SZ", "OZ"}}};
  for (uint32_t c = 0; c < 3; ++c) {
    a.Alu(vp::Salu::kAdd, 29, 18, 0, 8 * c);
    a.Memory(vp::Smem::kSld, {.rd = 14, .ra = 8, .rb = 29});
    k.Pl(11, kFixed[c].first);
    k.Pl(12, kFixed[c].second);
    k.P({.rx = 11, .rd = 12, .rimm = 14, .b = vp::B::kImm});
  }
  a.Branch(vp::Cond::kAlways, 0, "rot");
  a.Label("learn");
  a.Alu(vp::Salu::kSub, 29, 18, 0, 7);
  k.Pl(21, "LQX");
  k.Pl(22, "LQY");
  k.Pl(23, "LQZ");
  a.Alu(vp::Salu::kAdd, 24, 2, 0, 1);
  a.Alu(vp::Salu::kAdd, 25, 2, 0, 2);
  for (uint32_t q = 0; q < learn; ++q) {
    a.Alu(vp::Salu::kSeq, 30, 29, 0, q);
    a.Branch(vp::Cond::kZ, 30, std::format("nq{}", q));
    for (const auto& [rx, rd] : {std::pair{2U, 21U}, {24U, 22U}, {25U, 23U}}) {
      k.P({.rx = rx, .rd = rd, .px = vp::Perm::kBcast, .pcol = q});
    }
    a.Branch(vp::Cond::kAlways, 0, "learn_c");
    a.Label(std::format("nq{}", q));
  }
  a.Label("learn_c");
  k.Ld(14, kConstNhalf);
  k.Ld(15, kConstSign);
  k.Ld(16, kConstZero);
  constexpr std::array<std::array<std::string_view, 3>, 3> kLearned = {
      {{"LQX", "SX", "OX"}, {"LQY", "SY", "OY"}, {"LQZ", "SZ", "OZ"}}};
  for (const auto& [lq, sz, off] : kLearned) {
    k.Pl(11, lq);
    k.Pl(12, "T1");
    k.Pl(13, off);
    k.Pl(21, sz);
    k.P({.rx = 11, .rd = 12, .nl = vp::Nl::kAct});
    k.P({.rx = 12, .rd = 12, .rimm = 14, .c0 = vp::C0::kImm});
    k.P({.rx = 12,
         .rx2 = 11,
         .rd = 12,
         .rimm = 15,
         .rimm2 = 16,
         .c0 = vp::C0::kImm,
         .cmp = vp::Cmp::kLt,
         .cmp_src = vp::CmpSrc::kX2Imm2,
         .ialu = vp::Ialu::kXor,
         .c0_pred = true});
    k.P({.rx = 12, .rx2 = 21, .rd = 13, .b = vp::B::kX2});
  }
  a.Label("rot");
  k.Pl(11, "OX");
  k.Pl(12, "OY");
  k.Pl(13, "T1");
  k.Pl(21, "T2");
  k.Pl(22, "KX");
  k.Pl(23, "KY");
  k.Pl(24, "KZ");
  k.Pl(25, "OZ");
  const auto bx = [&](uint32_t rx2, uint32_t rd, uint32_t col) {
    k.P({.rx = 1,
         .rx2 = rx2,
         .rd = rd,
         .b = vp::B::kX2,
         .px = vp::Perm::kBcast,
         .pcol = col});
  };
  const auto bc0 = [&](uint32_t rx2, uint32_t rd, uint32_t col) {
    k.P({.rx = 1,
         .rx2 = rx2,
         .rd = rd,
         .c0 = vp::C0::kX2,
         .px = vp::Perm::kBcast,
         .pcol = col});
  };
  bx(11, 13, 7);
  bx(12, 21, 6);
  k.P({.rx = 13, .rx2 = 21, .rd = 13, .c1 = vp::C1::kX2, .neg_c1 = true});
  bc0(13, 22, 0);
  bx(11, 13, 6);
  bx(12, 21, 7);
  k.P({.rx = 13, .rx2 = 21, .rd = 13, .c1 = vp::C1::kX2});
  bc0(13, 23, 1);
  bc0(25, 24, 2);
  k.Project(22, 23, 24, {"U", "V", "T3"});
  k.Location("T3");
  a.Ret();
  return a;
}

vp::Assembler DetKeypointLevel(uint32_t tq) {
  vp::Assembler a(true);
  Keypoint k(a, tq, false);
  k.Prologue();
  a.Alu(vp::Salu::kShl, 29, 18, 0, 2);
  a.Memory(vp::Smem::kSld, {.rd = 14, .ra = 9, .rb = 29});
  a.Alu(vp::Salu::kAdd, 29, 29, 0, 1);
  a.Memory(vp::Smem::kSld, {.rd = 15, .ra = 9, .rb = 29});
  k.Pl(11, "VM");
  a.Emit({.rx = 11,
          .rq = 4,
          .rnw = 28,
          .rn8 = 5,
          .wr = vp::Write::kNone,
          .ialu = vp::Ialu::kOr,
          .rm = vp::RowMask::kSet});
  k.P({.rx = 11,
       .rd = 3,
       .cmp = vp::Cmp::kLt,
       .cmp_src = vp::CmpSrc::kLane,
       .cmp_int = true,
       .sel = vp::Sel::kNull,
       .dst_s1 = true});
  a.SkipMaskedRows(true);
  k.HalfPixel();
  k.Pl(23, "P1");
  k.Pl(24, "P2");
  k.Pl(25, "T3");
  k.Ld(16, kConstCf);
  k.Ld(17, kConstNcf);
  k.Ld(26, kConstNone);
  k.Floors(26);
  k.Pl(26, "LH");
  k.Pl(27, "LW");
  k.P({.rx = 11, .rx2 = 23, .rd = 26, .c1 = vp::C1::kX2, .neg_c1 = true});
  k.P({.rx = 12, .rx2 = 24, .rd = 27, .c1 = vp::C1::kX2, .neg_c1 = true});
  a.Alu(vp::Salu::kAdd, 29, 6, 0, 1);
  k.P({.rx = 23,
       .rd = 23,
       .rpb = 6,
       .pa = vp::Pbuf::kRowVec,
       .c0 = vp::C0::kPbuf});
  k.P({.rx = 24,
       .rd = 24,
       .rpb = 29,
       .pa = vp::Pbuf::kRowVec,
       .c0 = vp::C0::kPbuf});
  k.Pl(11, "T1");
  k.Pl(12, "T2");
  k.Pl(13, "DHP");
  k.Pl(25, "DWP");
  k.Pl(30, "P3");
  k.Ld(16, kConstOne);
  k.Ld(17, kConstNone);
  k.Ld(31, kConstZero);
  const auto select = [&](uint32_t rx, uint32_t rx2, uint32_t rd,
                          uint32_t bound) {
    k.P({.rx = rx,
         .rx2 = rx2,
         .rd = rd,
         .rimm2 = bound,
         .cmp = vp::Cmp::kEq,
         .cmp_src = vp::CmpSrc::kX2Imm2,
         .sel = vp::Sel::kZero});
  };
  select(26, 13, 11, 16);
  k.P({.rx = 26, .rd = 12, .rimm = 17, .c0 = vp::C0::kImm, .b = vp::B::kImm});
  select(12, 13, 12, 31);
  k.P({.rx = 11, .rx2 = 12, .rd = 11, .c0 = vp::C0::kX2});
  select(27, 25, 12, 16);
  k.P({.rx = 27, .rd = 30, .rimm = 17, .c0 = vp::C0::kImm, .b = vp::B::kImm});
  select(30, 25, 30, 31);
  k.P({.rx = 12, .rx2 = 30, .rd = 12, .c0 = vp::C0::kX2});
  k.P({.rx = 11, .rx2 = 12, .rd = 30, .b = vp::B::kX2});
  k.Pl(12, "VM");
  k.Pl(13, "IM");
  k.Inside(12, 13, 23, 31, vp::Cmp::kGe);
  k.Inside(13, 13, 23, 14, vp::Cmp::kLt);
  k.Inside(13, 13, 24, 31, vp::Cmp::kGe);
  k.Inside(13, 13, 24, 15, vp::Cmp::kLt);
  a.Alu(vp::Salu::kShl, 29, 18, 0, 2);
  a.Alu(vp::Salu::kAdd, 29, 29, 0, 2);
  a.Memory(vp::Smem::kSld, {.rd = 26, .ra = 9, .rb = 29});
  a.Alu(vp::Salu::kAdd, 29, 29, 0, 1);
  a.Memory(vp::Smem::kSld, {.rd = 27, .ra = 9, .rb = 29});
  k.Pl(11, "T1");
  k.Pl(25, "T3");
  k.Ld(16, kConstCf);
  k.Ld(17, kConstI2f0);
  k.P({.rx = 24, .rd = 11, .rimm = 14, .b = vp::B::kImm});
  k.P({.rx = 11, .rx2 = 23, .rd = 11, .c0 = vp::C0::kX2});
  k.P({.rx = 11, .rd = 11, .rimm = 27, .b = vp::B::kImm});
  a.RoundToInt(11, 11, 4, 28, 5, 17, 16, 25);
  k.P({.rx = 11,
       .rd = 11,
       .rimm = 26,
       .c0 = vp::C0::kImm,
       .ialu = vp::Ialu::kAdd});
  k.Ld(16, kConstOne);
  k.P({.rx = 11,
       .rx2 = 13,
       .rd = 11,
       .rimm2 = 16,
       .cmp = vp::Cmp::kEq,
       .cmp_src = vp::CmpSrc::kX2Imm2,
       .sel = vp::Sel::kNull});
  select(30, 13, 30, 16);
  a.Alu(vp::Salu::kAdd, 29, 6, 0, 2);
  a.Alu(vp::Salu::kAdd, 31, 6, 0, 3);
  k.P({.rx = 11,
       .rd = 11,
       .rpb = 29,
       .pa = vp::Pbuf::kRowVec,
       .c0 = vp::C0::kPbuf,
       .ialu = vp::Ialu::kAnd});
  k.P({.rx = 30,
       .rd = 25,
       .rpb = 31,
       .pa = vp::Pbuf::kRowVec,
       .c0 = vp::C0::kPbuf,
       .ialu = vp::Ialu::kAnd});
  k.P({.rx = 11,
       .rx2 = 25,
       .rd = 3,
       .c0 = vp::C0::kX2,
       .ialu = vp::Ialu::kOr,
       .dst_s1 = true});
  a.SkipMaskedRows(false);
  a.Ret();
  return a;
}

vp::Assembler MapKeypointInit(uint32_t tq) {
  vp::Assembler a(true);
  Keypoint k(a, tq, true);
  k.Prologue();
  k.Pl(11, "ONES");
  k.Ld(14, kConstOne);
  k.P({.rx = 1,
       .rd = 11,
       .rimm = 0,
       .c0 = vp::C0::kImm,
       .ialu = vp::Ialu::kAnd});
  k.P({.rx = 11, .rd = 11, .rimm = 14, .c0 = vp::C0::kImm});
  for (uint32_t c = 0; c < 2; ++c) {
    for (uint32_t gs = 0; gs < kMapGroupsPerBlock; ++gs) {
      k.Pl(12, std::format("ANC{}{}", c == 0 ? 'X' : 'Y', gs));
      for (uint32_t i = 0; i < 8; ++i) {
        const uint32_t s = (8 * gs) + i;
        if (s >= kMapSamples) {
          break;
        }
        const uint32_t col = (2 * s) + c;
        a.Alu(vp::Salu::kAdd, 13, 1, 0, col / 8);
        a.LoadImmediate(15, i);
        if (i == 0) {
          k.P({.rx = 13,
               .rd = 12,
               .rimm2 = 15,
               .px = vp::Perm::kBcast,
               .pcol = col % 8,
               .cmp = vp::Cmp::kEq,
               .cmp_src = vp::CmpSrc::kLane,
               .sel = vp::Sel::kZero});
        } else {
          k.P({.rx = 13,
               .rx2 = 12,
               .rd = 12,
               .rimm2 = 15,
               .px = vp::Perm::kBcast,
               .pcol = col % 8,
               .cmp = vp::Cmp::kEq,
               .cmp_src = vp::CmpSrc::kLane,
               .sel = vp::Sel::kX2});
        }
      }
    }
  }
  a.Ret();
  return a;
}

vp::Assembler MapKeypointPoint(uint32_t tq) {
  vp::Assembler a(true);
  Keypoint k(a, tq, true);
  k.Prologue();
  a.Alu(vp::Salu::kAdd, 11, 10, 0, k.PlaneOffset("ANCX0"));
  a.Alu(vp::Salu::kAdd, 11, 11, 22);
  a.Alu(vp::Salu::kAdd, 12, 10, 0, k.PlaneOffset("ANCY0"));
  a.Alu(vp::Salu::kAdd, 12, 12, 22);
  a.Move(13, 2);
  a.Alu(vp::Salu::kAdd, 13, 13, 22);
  a.Alu(vp::Salu::kAdd, 23, 2, 0, kMapGroupsPerBlock);
  a.Alu(vp::Salu::kAdd, 23, 23, 22);
  k.Pl(24, "KX");
  k.Pl(25, "KY");
  k.Pl(26, "KZ");
  k.Pl(27, "ONES");
  k.P({.rx = 11, .rx2 = 13, .rd = 24, .c0 = vp::C0::kX2});
  k.P({.rx = 12, .rx2 = 23, .rd = 25, .c0 = vp::C0::kX2});
  a.Memory(vp::Smem::kSld, {.rd = 14, .ra = 19, .rb = 20});
  k.P({.rx = 27, .rd = 26, .rimm = 14, .b = vp::B::kImm});
  k.Project(24, 25, 26, {"U", "V", "W"});
  k.Location("W");
  a.Ret();
  return a;
}

namespace {

void MapKeypointCorner(vp::Assembler& a, Keypoint& k, uint32_t n) {
  const bool dh = (n >> 1) != 0;
  const bool dw = (n & 1U) != 0;
  k.Pl(11, "HI");
  k.Pl(12, "WI");
  k.Pl(25, "T3");
  k.Pl(29, "IM");
  if (dh) {
    k.P({.rx = 23, .rd = 11, .rimm = 16, .c0 = vp::C0::kImm});
  } else {
    k.P({.rx = 23, .rd = 11});
  }
  if (dw) {
    k.P({.rx = 24, .rd = 12, .rimm = 16, .c0 = vp::C0::kImm});
  } else {
    k.P({.rx = 24, .rd = 12});
  }
  k.Pl(13, dh ? "LH" : "LH1");
  k.Pl(31, dw ? "LW" : "LW1");
  k.Pl(21, std::format("B{}", n));
  k.P({.rx = 13, .rx2 = 31, .rd = 21, .b = vp::B::kX2});
  k.Pl(13, "VM");
  k.Ld(31, kConstZero);
  k.Inside(13, 29, 11, 31, vp::Cmp::kGe);
  k.Inside(29, 29, 11, 14, vp::Cmp::kLt);
  k.Inside(29, 29, 12, 31, vp::Cmp::kGe);
  k.Inside(29, 29, 12, 15, vp::Cmp::kLt);
  k.Pl(13, std::format("A{}", n));
  k.Ld(30, kConstCf);
  k.P({.rx = 12, .rd = 13, .rimm = 14, .b = vp::B::kImm});
  k.P({.rx = 13, .rx2 = 11, .rd = 13, .c0 = vp::C0::kX2});
  k.P({.rx = 13, .rd = 13, .rimm = 27, .b = vp::B::kImm});
  a.RoundToInt(13, 13, 4, 28, 5, 17, 30, 25);
  k.P({.rx = 13,
       .rd = 13,
       .rimm = 26,
       .c0 = vp::C0::kImm,
       .ialu = vp::Ialu::kAdd});
  k.P({.rx = 13,
       .rx2 = 29,
       .rd = 13,
       .rimm2 = 16,
       .cmp = vp::Cmp::kEq,
       .cmp_src = vp::CmpSrc::kX2Imm2,
       .sel = vp::Sel::kNull});
  k.P({.rx = 21,
       .rx2 = 29,
       .rd = 21,
       .rimm2 = 16,
       .cmp = vp::Cmp::kEq,
       .cmp_src = vp::CmpSrc::kX2Imm2,
       .sel = vp::Sel::kZero});
}

}  // namespace

vp::Assembler MapKeypointLevel(uint32_t tq) {
  vp::Assembler a(true);
  Keypoint k(a, tq, true);
  k.Prologue();
  a.Alu(vp::Salu::kShl, 29, 18, 0, 2);
  for (const uint32_t rd : {14U, 15U, 26U, 27U}) {
    if (rd != 14) {
      a.Alu(vp::Salu::kAdd, 29, 29, 0, 1);
    }
    a.Memory(vp::Smem::kSld, {.rd = rd, .ra = 9, .rb = 29});
  }
  k.Pl(11, "VM");
  a.Emit({.rx = 11,
          .rq = 4,
          .rnw = 28,
          .rn8 = 5,
          .wr = vp::Write::kNone,
          .ialu = vp::Ialu::kOr,
          .rm = vp::RowMask::kSet});
  a.LoadImmediate(16, 4);
  for (const auto& [plane, sel] :
       {std::pair{"A0", vp::Sel::kNull}, std::pair{"B0", vp::Sel::kZero}}) {
    k.Pl(12, plane);
    a.Emit({.rx = 11,
            .rd = 12,
            .rq = 4,
            .rnw = 16,
            .rn8 = 5,
            .wr = vp::Write::kFbuf,
            .cmp = vp::Cmp::kLt,
            .cmp_src = vp::CmpSrc::kLane,
            .cmp_int = true,
            .sel = sel});
  }
  a.SkipMaskedRows(true);
  k.HalfPixel();
  k.Pl(23, "P1");
  k.Pl(24, "P2");
  k.Pl(25, "T3");
  k.Ld(16, kConstCf);
  k.Ld(17, kConstNcf);
  k.Ld(30, kConstNone);
  k.Floors(30);
  k.Pl(13, "LH");
  k.Pl(31, "LW");
  k.P({.rx = 11, .rx2 = 23, .rd = 13, .c1 = vp::C1::kX2, .neg_c1 = true});
  k.P({.rx = 12, .rx2 = 24, .rd = 31, .c1 = vp::C1::kX2, .neg_c1 = true});
  k.Pl(11, "LH1");
  k.Pl(12, "LW1");
  k.P({.rx = 13, .rd = 11, .rimm = 30, .c0 = vp::C0::kImm, .b = vp::B::kImm});
  k.P({.rx = 31, .rd = 12, .rimm = 30, .c0 = vp::C0::kImm, .b = vp::B::kImm});
  k.Ld(16, kConstOne);
  k.Ld(17, kConstI2f0);
  k.Ld(30, kConstCf);
  for (uint32_t n = 0; n < 4; ++n) {
    MapKeypointCorner(a, k, n);
  }
  a.SkipMaskedRows(false);
  k.Pl(11, "A0");
  a.LoadImmediate(12, 1);
  a.Transpose8(11, 3, 4, 5, 12, 12, 4, vp::TransposePack::kZip4);
  a.Ret();
  return a;
}

vp::Assembler InstanceBankKernel(uint32_t nap, uint32_t na, uint32_t ntemp,
                                 BankMode mode, uint32_t k) {
  vp::Assembler a(true);
  const uint32_t nw = nap / 32;
  const bool cache = mode == BankMode::kCache;
  const uint32_t off0 = mode == BankMode::kUpdate ? ntemp : 0;
  a.LoadImmediate(11, 1);
  a.LoadImmediate(12, 4);
  a.LoadImmediate(9, nw);
  const uint32_t conf = 0;
  const uint32_t idx = nw;
  const uint32_t rank = 2 * nw;
  const uint32_t tmp = 3 * nw;
  const uint32_t entx = 4 * nw;
  const uint32_t enta = 5 * nw;
  const uint32_t ones = 6 * nw;
  const uint32_t tmpw = (6 * nw) + 1;
  const uint32_t lanew = (6 * nw) + 2;
  const auto w = [&](uint32_t reg, uint32_t off) {
    a.Alu(vp::Salu::kAdd, reg, 2, 0, off);
  };
  const auto ld = [&](uint32_t rd, uint32_t column) {
    a.Memory(vp::Smem::kSld, {.rd = rd, .ra = 8, .col = column});
  };
  const auto pp = [&](Pass pass) {
    pass.rq = 11;
    pass.rnw = 9;
    pass.rn8 = 9;
    pass.wr = vp::Write::kFbuf;
    a.Emit(pass);
  };
  w(13, ones);
  ld(14, kIkOne);
  a.Emit({.rx = 13,
          .rd = 13,
          .rq = 11,
          .rnw = 11,
          .rn8 = 11,
          .rimm = 0,
          .c0 = vp::C0::kImm,
          .wr = vp::Write::kFbuf,
          .ialu = vp::Ialu::kAnd});
  a.Emit({.rx = 13,
          .rd = 13,
          .rq = 11,
          .rnw = 11,
          .rn8 = 11,
          .rimm = 14,
          .c0 = vp::C0::kImm,
          .wr = vp::Write::kFbuf});
  a.LoadImmediate(14, 0);
  a.Move(15, 1);
  w(16, conf);
  w(17, tmpw);
  a.Label("rm");
  a.Emit({.rx = 15,
          .rq = 11,
          .rnw = 12,
          .rn8 = 12,
          .rimm2 = 5,
          .mask_en = true,
          .red = vp::Reduce::kMax,
          .sc = vp::RowScalar::kMax});
  a.Alu(vp::Salu::kAnd, 18, 14, 0, 7);
  a.Emit({.rx = 13,
          .rd = 17,
          .rq = 11,
          .rnw = 11,
          .rn8 = 11,
          .rimm2 = 18,
          .b = vp::B::kRs1,
          .wr = vp::Write::kFbuf,
          .cmp = vp::Cmp::kEq,
          .cmp_src = vp::CmpSrc::kLane,
          .sel = vp::Sel::kZero});
  a.Alu(vp::Salu::kSeq, 19, 18, 0, 0);
  a.Branch(vp::Cond::kZ, 19, "rm_acc");
  a.Emit({.rx = 17,
          .rd = 16,
          .rq = 11,
          .rnw = 11,
          .rn8 = 11,
          .wr = vp::Write::kFbuf,
          .px = vp::Perm::kT4});
  a.Branch(vp::Cond::kAlways, 0, "rm_next");
  a.Label("rm_acc");
  a.Emit({.rx = 17,
          .rx2 = 16,
          .rd = 16,
          .rq = 11,
          .rnw = 11,
          .rn8 = 11,
          .c0 = vp::C0::kX2,
          .wr = vp::Write::kFbuf,
          .px = vp::Perm::kT4,
          .ialu = vp::Ialu::kOr});
  a.Label("rm_next");
  a.Alu(vp::Salu::kAdd, 15, 15, 12);
  a.Alu(vp::Salu::kAdd, 14, 14, 0, 1);
  a.Alu(vp::Salu::kSeq, 19, 18, 0, 7);
  a.Branch(vp::Cond::kZ, 19, "rm_skip");
  a.Alu(vp::Salu::kAdd, 16, 16, 0, 1);
  a.Label("rm_skip");
  a.Alu(vp::Salu::kSub, 19, 4, 14);
  a.Branch(vp::Cond::kNz, 19, "rm");
  w(16, conf);
  w(17, idx);
  w(20, tmp);
  if (cache) {
    ld(14, kIkNhalf);
    ld(15, kIkSign);
    ld(21, kIkHalf);
    ld(22, kIkZero);
    pp({.rx = 16, .rd = 20, .nl = vp::Nl::kAct});
    pp({.rx = 20, .rd = 20, .rimm = 14, .c0 = vp::C0::kImm});
    pp({.rx = 20,
        .rx2 = 16,
        .rd = 20,
        .rimm = 15,
        .rimm2 = 22,
        .c0 = vp::C0::kImm,
        .cmp = vp::Cmp::kLt,
        .cmp_src = vp::CmpSrc::kX2Imm2,
        .ialu = vp::Ialu::kXor,
        .c0_pred = true});
    pp({.rx = 20, .rd = 16, .rimm = 21, .c0 = vp::C0::kImm});
    a.Alu(vp::Salu::kAdd, 21, 3, 0, 3 * nw);
    ld(14, kIkDecay);
    pp({.rx = 21, .rd = 20, .rimm = 14, .b = vp::B::kImm});
    pp({.rx = 16, .rx2 = 20, .rd = 16, .c1 = vp::C1::kMax});
  }
  ld(14, kIkFna);
  ld(15, kIkNinf);
  pp({.rx = 16,
      .rx2 = 17,
      .rd = 16,
      .rimm = 15,
      .rimm2 = 14,
      .cmp = vp::Cmp::kLt,
      .cmp_src = vp::CmpSrc::kX2Imm2,
      .sel = vp::Sel::kImm});
  w(18, entx);
  w(19, enta);
  ld(21, kIkI2f0);
  ld(22, kIkCf);
  for (const auto& [ent, tb, base] :
       {std::array{18U, kIkTbX, kIkBaseX}, std::array{19U, kIkTbA, kIkBaseA}}) {
    ld(14, tb);
    pp({.rx = 17, .rd = ent, .rimm = 14, .b = vp::B::kImm});
    a.RoundToInt(ent, ent, 11, 9, 9, 21, 22, 20);
    ld(14, base);
    pp({.rx = ent,
        .rd = ent,
        .rimm = 14,
        .c0 = vp::C0::kImm,
        .ialu = vp::Ialu::kAdd});
  }
  w(23, rank);
  pp({.rx = 16,
      .rd = 23,
      .rimm = 0,
      .c0 = vp::C0::kImm,
      .ialu = vp::Ialu::kAnd});
  ld(24, kIkOne);
  ld(25, kIkNone);
  a.LoadImmediate(26, 0);
  a.Label("rk");
  a.Alu(vp::Salu::kShr, 27, 26, 0, 5);
  a.Alu(vp::Salu::kAdd, 28, 27, 16);
  a.Alu(vp::Salu::kAdd, 27, 27, 17);
  a.Memory(vp::Smem::kSld, {.rd = 29, .ra = 28, .rb = 26});
  a.Memory(vp::Smem::kSld, {.rd = 30, .ra = 27, .rb = 26});
  pp({.rx = 23,
      .rx2 = 16,
      .rd = 23,
      .rimm = 24,
      .rimm2 = 29,
      .c0 = vp::C0::kImm,
      .cmp = vp::Cmp::kLt,
      .cmp_src = vp::CmpSrc::kX2Imm2,
      .c0_pred = true});
  pp({.rx = 17,
      .rx2 = 16,
      .rd = 20,
      .rimm = 25,
      .rimm2 = 29,
      .cmp = vp::Cmp::kEq,
      .cmp_src = vp::CmpSrc::kX2Imm2,
      .sel = vp::Sel::kImm});
  pp({.rx = 23,
      .rx2 = 20,
      .rd = 23,
      .rimm = 24,
      .rimm2 = 30,
      .c0 = vp::C0::kImm,
      .cmp = vp::Cmp::kGt,
      .cmp_src = vp::CmpSrc::kX2Imm2,
      .c0_pred = true});
  a.Alu(vp::Salu::kAdd, 26, 26, 0, 1);
  a.Alu(vp::Salu::kSub, 31, 26, 0, na);
  a.Branch(vp::Cond::kNz, 31, "rk");
  if (cache) {
    a.Alu(vp::Salu::kAdd, 28, 3, 0, nw);
    ld(14, kIkNinf);
    pp({.rx = 16,
        .rd = 28,
        .rimm = 0,
        .c0 = vp::C0::kImm,
        .ialu = vp::Ialu::kAnd});
    pp({.rx = 28,
        .rd = 28,
        .rimm = 14,
        .c0 = vp::C0::kImm,
        .ialu = vp::Ialu::kOr});
  }
  w(13, lanew);
  a.LoadImmediate(26, 0);
  std::vector<std::pair<uint32_t, uint32_t>> outs = {{18, 0}, {19, nw}};
  if (cache) {
    outs = {{18, 0}, {16, nw}, {17, 2 * nw}, {19, 4 * nw}};
  }
  a.Label("sel");
  a.Alu(vp::Salu::kShr, 27, 26, 0, 5);
  a.Alu(vp::Salu::kAdd, 27, 27, 17);
  a.Memory(vp::Smem::kSld, {.rd = 30, .ra = 27, .rb = 26});
  for (const auto& [src, off] : outs) {
    pp({.rx = src,
        .rx2 = 23,
        .rd = 20,
        .rimm = 0,
        .rimm2 = 30,
        .c0 = vp::C0::kImm,
        .red = vp::Reduce::kMax,
        .sc = vp::RowScalar::kMax,
        .cmp = vp::Cmp::kEq,
        .cmp_src = vp::CmpSrc::kX2Imm2,
        .sel = vp::Sel::kZero,
        .ialu = vp::Ialu::kOr});
    a.Memory(vp::Smem::kRsmov, {.rd = 28, .col = 0});
    a.Memory(vp::Smem::kRsmov, {.rd = 29, .col = 1});
    a.Alu(vp::Salu::kOr, 28, 28, 29);
    a.Memory(vp::Smem::kRsmov, {.rd = 29, .col = 2});
    a.Alu(vp::Salu::kOr, 28, 28, 29);
    a.Memory(vp::Smem::kRsmov, {.rd = 29, .col = 3});
    a.Alu(vp::Salu::kOr, 28, 28, 29);
    a.Alu(vp::Salu::kAdd, 31, 26, 0, off0);
    a.Alu(vp::Salu::kShr, 27, 31, 0, 5);
    a.Alu(vp::Salu::kAdd, 27, 27, 3);
    if (off != 0) {
      a.Alu(vp::Salu::kAdd, 27, 27, 0, off);
    }
    a.Alu(vp::Salu::kAnd, 31, 31, 0, 31);
    a.StoreScalar(27, 13, 31, 28, 11);
  }
  a.Alu(vp::Salu::kAdd, 26, 26, 0, 1);
  a.Alu(vp::Salu::kSub, 31, 26, 0, k);
  a.Branch(vp::Cond::kNz, 31, "sel");
  a.Ret();
  return a;
}

vp::Assembler LongSoftmaxKernel() {
  vp::Assembler a;
  a.LoadImmediate(11, 1);
  a.Label("loop");
  a.Emit({.rx = 1,
          .rq = 11,
          .rnw = 8,
          .rn8 = 11,
          .red = vp::Reduce::kMax,
          .sc = vp::RowScalar::kNegMax});
  a.Emit({.rx = 1,
          .rd = 2,
          .rq = 11,
          .rnw = 8,
          .rn8 = 11,
          .c0 = vp::C0::kRs0,
          .b = vp::B::kLog2e,
          .nl = vp::Nl::kExp2,
          .red = vp::Reduce::kSum,
          .wr = vp::Write::kFbuf,
          .sc = vp::RowScalar::kRecipSum});
  a.Emit({.rx = 2,
          .rd = 3,
          .rq = 11,
          .rnw = 8,
          .rn8 = 11,
          .b = vp::B::kRs1,
          .wr = vp::Write::kFbuf});
  a.Alu(vp::Salu::kAdd, 1, 1, 8);
  a.Alu(vp::Salu::kAdd, 2, 2, 8);
  a.Alu(vp::Salu::kAdd, 3, 3, 8);
  a.Alu(vp::Salu::kSub, 4, 4, 0, 1);
  a.Branch(vp::Cond::kNz, 4, "loop");
  a.Ret();
  return a;
}

vp::Assembler EgoMeanKernel() {
  constexpr uint32_t kQuads = 11;
  vp::Assembler a(true);
  a.LoadImmediate(11, 1);
  a.LoadImmediate(12, 32);
  a.Emit({.rx = 1,
          .rd = 3,
          .rq = 11,
          .rnw = 12,
          .rn8 = 12,
          .wr = vp::Write::kFbuf});
  a.Move(13, 1);
  for (uint32_t q = 1; q < kQuads; ++q) {
    a.Alu(vp::Salu::kAdd, 13, 13, 0, 32);
    a.Emit({.rx = 13,
            .rx2 = 3,
            .rd = 3,
            .rq = 11,
            .rnw = 12,
            .rn8 = 12,
            .c0 = vp::C0::kX2,
            .wr = vp::Write::kFbuf});
  }
  for (const vp::Perm perm : {vp::Perm::kTswap1, vp::Perm::kTswap2}) {
    a.Emit({.rx = 3,
            .rx2 = 3,
            .rd = 3,
            .rq = 11,
            .rnw = 12,
            .rn8 = 12,
            .c0 = vp::C0::kX2,
            .wr = vp::Write::kFbuf,
            .px = perm});
  }
  a.Emit({.rx = 3,
          .rd = 3,
          .rq = 11,
          .rnw = 12,
          .rn8 = 12,
          .rimm = 8,
          .b = vp::B::kImm,
          .wr = vp::Write::kFbuf});
  a.Ret();
  return a;
}

vp::Assembler MotionClsIdx(uint32_t nap, uint32_t ncls) {
  vp::Assembler a(true);
  const uint32_t q = nap / 4;
  a.LoadImmediate(11, 1);
  a.LoadImmediate(12, 4);
  a.Move(13, 2);
  a.Alu(vp::Salu::kAdd, 14, 2, 0, 1);
  a.Alu(vp::Salu::kAdd, 15, 2, 0, 2);
  a.Alu(vp::Salu::kAdd, 16, 2, 0, 3);
  a.Alu(vp::Salu::kAdd, 17, 2, 0, 4 * q);
  a.Alu(vp::Salu::kAdd, 18, 2, 0, (4 * q) + 1);
  const auto ld = [&](uint32_t rd, uint32_t column) {
    a.Memory(vp::Smem::kSld, {.rd = rd, .ra = 8, .col = column});
  };
  ld(19, kIkOne);
  a.Emit({.rx = 18,
          .rd = 18,
          .rq = 11,
          .rnw = 11,
          .rn8 = 11,
          .rimm = 0,
          .c0 = vp::C0::kImm,
          .wr = vp::Write::kFbuf,
          .ialu = vp::Ialu::kAnd});
  a.Emit({.rx = 18,
          .rd = 18,
          .rq = 11,
          .rnw = 11,
          .rn8 = 11,
          .rimm = 19,
          .c0 = vp::C0::kImm,
          .wr = vp::Write::kFbuf});
  a.LoadImmediate(20, 0);
  a.Move(21, 1);
  a.Move(22, 13);
  a.Label("mx");
  a.Emit({.rx = 21,
          .rq = 11,
          .rnw = 12,
          .rn8 = 12,
          .rimm2 = 5,
          .mask_en = true,
          .red = vp::Reduce::kMax,
          .sc = vp::RowScalar::kMax});
  a.Emit({.rx = 18,
          .rd = 22,
          .rq = 11,
          .rnw = 11,
          .rn8 = 11,
          .b = vp::B::kRs1,
          .wr = vp::Write::kFbuf});
  a.Alu(vp::Salu::kAdd, 21, 21, 12);
  a.Alu(vp::Salu::kAdd, 22, 22, 12);
  a.Alu(vp::Salu::kAdd, 20, 20, 0, 1);
  a.Alu(vp::Salu::kSub, 23, 4, 20);
  a.Branch(vp::Cond::kNz, 23, "mx");
  const auto p = [&](Pass pass) {
    pass.rq = 4;
    pass.rnw = 11;
    pass.rn8 = 12;
    pass.wr = vp::Write::kFbuf;
    a.Emit(pass);
  };
  ld(19, kIkNone);
  p({.rx = 14,
     .rd = 14,
     .rimm = 0,
     .c0 = vp::C0::kImm,
     .ialu = vp::Ialu::kAnd});
  p({.rx = 14, .rd = 14, .rimm = 19, .c0 = vp::C0::kImm});
  ld(23, kIkZero);
  for (uint32_t c = ncls; c-- > 0;) {
    a.Alu(vp::Salu::kAdd, 21, 1, 0, c / 8);
    p({.rx = 21,
       .rx2 = 13,
       .rd = 15,
       .c1 = vp::C1::kX2,
       .neg_c1 = true,
       .px = vp::Perm::kBcast,
       .pcol = c % 8});
    a.LoadImmediate(24, FloatBits(c));
    p({.rx = 14,
       .rx2 = 15,
       .rd = 14,
       .rimm = 24,
       .rimm2 = 23,
       .cmp = vp::Cmp::kNe,
       .cmp_src = vp::CmpSrc::kX2Imm2,
       .sel = vp::Sel::kImm});
  }
  ld(24, kIkTbX);
  p({.rx = 14, .rd = 16, .rimm = 24, .b = vp::B::kImm});
  ld(21, kIkI2f0);
  ld(22, kIkCf);
  a.RoundToInt(16, 16, 4, 11, 12, 21, 22, 15);
  ld(24, kIkBaseX);
  p({.rx = 16,
     .rd = 16,
     .rimm = 24,
     .c0 = vp::C0::kImm,
     .ialu = vp::Ialu::kAdd});
  a.LoadImmediate(20, 0);
  a.Move(21, 16);
  a.Move(22, 3);
  a.Label("tp");
  a.Alu(vp::Salu::kAnd, 23, 20, 0, 7);
  a.Emit({.rx = 21,
          .rd = 17,
          .rq = 11,
          .rnw = 11,
          .rn8 = 11,
          .rimm2 = 23,
          .wr = vp::Write::kFbuf,
          .cmp = vp::Cmp::kEq,
          .cmp_src = vp::CmpSrc::kLane,
          .sel = vp::Sel::kZero});
  a.Alu(vp::Salu::kSeq, 24, 23, 0, 0);
  a.Branch(vp::Cond::kZ, 24, "tp_acc");
  a.Emit({.rx = 17,
          .rd = 22,
          .rq = 11,
          .rnw = 11,
          .rn8 = 11,
          .wr = vp::Write::kFbuf,
          .px = vp::Perm::kT4});
  a.Branch(vp::Cond::kAlways, 0, "tp_next");
  a.Label("tp_acc");
  a.Emit({.rx = 17,
          .rx2 = 22,
          .rd = 22,
          .rq = 11,
          .rnw = 11,
          .rn8 = 11,
          .c0 = vp::C0::kX2,
          .wr = vp::Write::kFbuf,
          .px = vp::Perm::kT4,
          .ialu = vp::Ialu::kOr});
  a.Label("tp_next");
  a.Alu(vp::Salu::kAdd, 21, 21, 12);
  a.Alu(vp::Salu::kAdd, 20, 20, 0, 1);
  a.Alu(vp::Salu::kSeq, 24, 23, 0, 7);
  a.Branch(vp::Cond::kZ, 24, "tp_skip");
  a.Alu(vp::Salu::kAdd, 22, 22, 0, 1);
  a.Label("tp_skip");
  a.Alu(vp::Salu::kSub, 24, 4, 20);
  a.Branch(vp::Cond::kNz, 24, "tp");
  a.Ret();
  return a;
}

vp::Assembler MotionRot() {
  vp::Assembler a(true);
  a.LoadImmediate(11, 1);
  a.LoadImmediate(12, 4);
  const auto p = [&](Pass pass) {
    pass.rq = 4;
    pass.rnw = 11;
    pass.rn8 = 12;
    pass.wr = vp::Write::kFbuf;
    a.Emit(pass);
  };
  constexpr uint32_t kRx = 3;
  constexpr uint32_t kRy = 13;
  constexpr uint32_t kT1 = 14;
  constexpr uint32_t kT2 = 15;
  constexpr uint32_t kTabY = 16;
  constexpr uint32_t kA = 17;
  constexpr uint32_t kY = 18;
  constexpr uint32_t kC = 19;
  constexpr uint32_t kS = 20;
  for (const auto& [rd, ra, imm] :
       {std::array{13U, 3U, 1U}, std::array{14U, 3U, 2U},
        std::array{15U, 3U, 3U}, std::array{16U, 2U, 1U},
        std::array{17U, 1U, 2U}, std::array{18U, 1U, 3U},
        std::array{19U, 2U, 2U}, std::array{20U, 2U, 3U}}) {
    a.Alu(vp::Salu::kAdd, rd, ra, 0, imm);
  }
  a.LoadImmediate(21, FloatBits(1));
  a.LoadImmediate(22, 0x5F3759DFU);
  a.LoadImmediate(23, Decimal(-1, 2));
  a.LoadImmediate(24, Decimal(3, 2));
  p({.rx = 1, .rd = kT1, .b = vp::B::kPre, .px = vp::Perm::kBcast, .pcol = 7});
  p({.rx = 1, .rd = kT2, .b = vp::B::kPre, .px = vp::Perm::kBcast, .pcol = 6});
  p({.rx = kT1, .rx2 = kT2, .rd = kA, .c0 = vp::C0::kX2});
  p({.rx = kA,
     .rd = kT2,
     .rimm = 21,
     .rimm2 = 0,
     .c0 = vp::C0::kZero,
     .cmp = vp::Cmp::kGt,
     .cmp_src = vp::CmpSrc::kImm2,
     .sel = vp::Sel::kImm,
     .ialu = vp::Ialu::kAnd});
  p({.rx = kA,
     .rd = kA,
     .rimm = 21,
     .rimm2 = 0,
     .cmp = vp::Cmp::kGt,
     .cmp_src = vp::CmpSrc::kImm2,
     .sel = vp::Sel::kImm});
  p({.rx = kA, .rd = kY, .ialu = vp::Ialu::kShr1});
  p({.rx = kY,
     .rd = kY,
     .rimm = 22,
     .c0 = vp::C0::kImm,
     .ialu = vp::Ialu::kRsub});
  for (int i = 0; i < 3; ++i) {
    p({.rx = kY, .rd = kT1, .b = vp::B::kPre});
    p({.rx = kT1, .rx2 = kA, .rd = kT1, .b = vp::B::kX2});
    p({.rx = kT1, .rd = kT1, .rimm = 23, .b = vp::B::kImm});
    p({.rx = kT1, .rd = kT1, .rimm = 24, .c0 = vp::C0::kImm});
    p({.rx = kY, .rx2 = kT1, .rd = kY, .b = vp::B::kX2});
  }
  p({.rx = 1,
     .rx2 = kY,
     .rd = kC,
     .b = vp::B::kX2,
     .px = vp::Perm::kBcast,
     .pcol = 7});
  p({.rx = 1,
     .rx2 = kY,
     .rd = kS,
     .b = vp::B::kX2,
     .px = vp::Perm::kBcast,
     .pcol = 6});
  p({.rx = kC, .rx2 = kT2, .rd = kC, .c0 = vp::C0::kX2});
  p({.rx = kC, .rx2 = 2, .rd = kT1, .b = vp::B::kX2});
  p({.rx = kS, .rx2 = kTabY, .rd = kT2, .b = vp::B::kX2});
  p({.rx = kT1, .rx2 = kT2, .rd = kRx, .c1 = vp::C1::kX2, .neg_c1 = true});
  p({.rx = kS, .rx2 = 2, .rd = kT1, .b = vp::B::kX2});
  p({.rx = kC, .rx2 = kTabY, .rd = kT2, .b = vp::B::kX2});
  p({.rx = kT1, .rx2 = kT2, .rd = kRy, .c1 = vp::C1::kX2});
  a.Ret();
  return a;
}

vp::Assembler MotionBcast(uint32_t modes) {
  vp::Assembler a(true);
  a.LoadImmediate(11, 1);
  a.LoadImmediate(12, 4);
  a.Alu(vp::Salu::kAdd, 13, 1, 0, 1);
  a.Alu(vp::Salu::kShl, 14, 4, 0, 2);
  a.Alu(vp::Salu::kAdd, 14, 14, 3);
  for (uint32_t m = 0; m < modes; ++m) {
    a.Alu(vp::Salu::kSeq, 15, 5, 0, m);
    a.Branch(vp::Cond::kZ, 15, std::format("nm{}", m));
    for (uint32_t k = 0; k < 4; ++k) {
      for (const auto& [base, rx] : {std::pair{3U, 1U}, {14U, 13U}}) {
        a.Alu(vp::Salu::kAdd, 16, base, 0, k);
        a.Emit({.rx = rx,
                .rd = 16,
                .rq = 4,
                .rnw = 11,
                .rn8 = 12,
                .wr = vp::Write::kFbuf,
                .px = vp::Perm::kBcast,
                .pcol = m});
      }
    }
    a.Ret();
    a.Label(std::format("nm{}", m));
  }
  a.Ret();
  return a;
}

vp::Assembler MotionCompact() {
  vp::Assembler a(true);
  a.LoadImmediate(11, 1);
  a.LoadImmediate(12, 4);
  a.Emit({.rx = 1,
          .rd = 3,
          .rq = 4,
          .rnw = 11,
          .rn8 = 12,
          .wr = vp::Write::kFbuf,
          .dst_s1 = true});
  a.Ret();
  return a;
}

vp::Assembler MotionEmbed() {
  vp::Assembler a(true);
  a.LoadImmediate(11, 1);
  a.LoadImmediate(12, 32);
  for (uint32_t g = 0; g < 32; ++g) {
    const uint32_t src = g < 16 ? 1 : 2;
    const uint32_t gg = g % 16;
    a.Alu(vp::Salu::kAdd, 14, 6, 0, gg);
    a.Alu(vp::Salu::kAdd, 15, 6, 0, 16 + gg);
    a.Alu(vp::Salu::kAdd, 16, 3, 0, g);
    a.Emit({.rx = src,
            .rd = 16,
            .rq = 4,
            .rnw = 11,
            .rn8 = 12,
            .rpb = 14,
            .pa = vp::Pbuf::kRowVec,
            .pa_at_mul = true,
            .b = vp::B::kPbuf,
            .wr = vp::Write::kFbuf,
            .x_s1 = true});
    a.Emit({.rx = 16,
            .rd = 16,
            .rq = 4,
            .rnw = 11,
            .rn8 = 12,
            .rpb = 15,
            .pa = vp::Pbuf::kRowVec,
            .c0 = vp::C0::kPbuf,
            .nl = vp::Nl::kTab01,
            .wr = vp::Write::kFbuf});
  }
  a.Ret();
  return a;
}

vp::Assembler AnchorProjectKernel(bool box, uint32_t words) {
  vp::Assembler a(true);
  a.LoadImmediate(11, 1);
  const auto lin = [&](uint32_t src, uint32_t dst, uint32_t pb, bool bias) {
    for (uint32_t k = 0; k < 8; ++k) {
      a.Alu(vp::Salu::kAdd, 14, 6, 0, pb + k);
      if (k == 0) {
        a.Emit({.rx = src,
                .rd = dst,
                .rq = 4,
                .rnw = 11,
                .rn8 = 5,
                .rpb = 14,
                .pa = vp::Pbuf::kRowVec,
                .pa_at_mul = true,
                .b = vp::B::kPbuf,
                .wr = vp::Write::kFbuf,
                .px = vp::Perm::kBcast,
                .pcol = 0});
      } else {
        a.Emit({.rx = src,
                .rx2 = dst,
                .rd = dst,
                .rq = 4,
                .rnw = 11,
                .rn8 = 5,
                .rpb = 14,
                .pa = vp::Pbuf::kRowVec,
                .pa_at_mul = true,
                .b = vp::B::kPbuf,
                .c1 = vp::C1::kX2,
                .wr = vp::Write::kFbuf,
                .px = vp::Perm::kBcast,
                .pcol = k});
      }
    }
    if (bias) {
      a.Alu(vp::Salu::kAdd, 14, 6, 0, pb + 8);
      a.Emit({.rx = dst,
              .rd = dst,
              .rq = 4,
              .rnw = 11,
              .rn8 = 5,
              .rpb = 14,
              .pa = vp::Pbuf::kRowVec,
              .c0 = vp::C0::kPbuf,
              .wr = vp::Write::kFbuf});
    }
  };
  if (box) {
    a.Memory(vp::Smem::kSld, {.rd = 8, .ra = 2, .col = 0});
    a.Alu(vp::Salu::kAdd, 12, 1, 0, 1);
    a.Alu(vp::Salu::kAdd, 13, 3, 0, 1);
    a.Alu(vp::Salu::kAdd, 16, 3, 0, 2);
    for (uint32_t k = 0; k < 3; ++k) {
      a.LoadImmediate(15, k);
      a.Emit({.rx = 12,
              .rx2 = k == 0 ? 1U : 16U,
              .rd = 16,
              .rq = 4,
              .rnw = 11,
              .rn8 = 5,
              .rimm = 8,
              .rimm2 = 15,
              .b = vp::B::kImm,
              .c1 = vp::C1::kX2,
              .wr = vp::Write::kFbuf,
              .px = vp::Perm::kBcast,
              .pcol = k,
              .cmp = vp::Cmp::kEq,
              .cmp_src = vp::CmpSrc::kLane,
              .sel = vp::Sel::kX2});
    }
    lin(16, 3, 0, true);
    lin(12, 13, 9, false);
    for (uint32_t w = 2; w < words; ++w) {
      a.Alu(vp::Salu::kAdd, 12, 1, 0, w);
      a.Alu(vp::Salu::kAdd, 13, 3, 0, w);
      a.Emit({.rx = 12,
              .rd = 13,
              .rq = 4,
              .rnw = 11,
              .rn8 = 5,
              .wr = vp::Write::kFbuf});
    }
  } else {
    for (uint32_t w = 0; w < words; ++w) {
      a.Alu(vp::Salu::kAdd, 12, 1, 0, w);
      a.Alu(vp::Salu::kAdd, 13, 3, 0, w);
      lin(12, 13, 0, true);
    }
  }
  a.Ret();
  return a;
}

namespace {

class FrameStart {
 public:
  explicit FrameStart(vp::Assembler& a) : a_(a) {
    for (uint32_t r = 12; r < 32; ++r) {
      pool_.push_back(r);
    }
  }

  uint32_t At(uint32_t offset) {
    return Reg({0, offset},
               [&](uint32_t r) { a_.Alu(vp::Salu::kAdd, r, 1, 0, offset); });
  }

  uint32_t Pb(uint32_t word) {
    return Reg({1, word},
               [&](uint32_t r) { a_.Alu(vp::Salu::kAdd, r, 6, 0, word); });
  }

  uint32_t K(uint32_t bits) {
    if (bits == 0) {
      return 0;
    }
    if (bits == 1) {
      return 11;
    }
    return Reg({2, bits}, [&](uint32_t r) { a_.LoadImmediate(r, bits); });
  }

  struct Operands {
    uint32_t rx = 0;
    uint32_t rd = 0;
    std::optional<uint32_t> rx2 = std::nullopt;
    uint32_t rq = 1;
    std::optional<uint32_t> imm = std::nullopt;
    std::optional<uint32_t> imm2 = std::nullopt;
  };

  void P(const Operands& o, Pass pass) {
    pass.rx = At(o.rx);
    pass.rd = At(o.rd);
    pass.rq = K(o.rq);
    pass.rnw = 11;
    pass.rn8 = 11;
    if (o.rx2.has_value()) {
      pass.rx2 = At(*o.rx2);
    }
    if (o.imm.has_value()) {
      pass.rimm = K(*o.imm);
    }
    if (o.imm2.has_value()) {
      pass.rimm2 = K(*o.imm2);
    }
    pass.wr = vp::Write::kFbuf;
    a_.Emit(pass);
  }

 private:
  using Key = std::pair<uint32_t, uint32_t>;

  template <typename F>
  uint32_t Reg(Key key, F emit) {
    const auto found = have_.find(key);
    uint32_t r = 0;
    if (found == have_.end()) {
      r = pool_.front();
      std::erase_if(have_,
                    [&](const auto& entry) { return entry.second == r; });
      emit(r);
      have_[key] = r;
    } else {
      r = found->second;
    }
    std::erase(pool_, r);
    pool_.push_back(r);
    return r;
  }

  vp::Assembler& a_;
  std::vector<uint32_t> pool_;
  std::map<Key, uint32_t> have_;
};

void Reciprocal(vp::Assembler& a, const std::array<uint32_t, 4>& planes,
                uint32_t rq, uint32_t r_magic, uint32_t r_none, uint32_t r_pb) {
  const auto [ra, ry, rt, ru] = planes;
  a.Emit({.rx = ra,
          .rd = ry,
          .rq = rq,
          .rnw = 11,
          .rn8 = 11,
          .rimm = r_magic,
          .c0 = vp::C0::kImm,
          .wr = vp::Write::kFbuf,
          .ialu = vp::Ialu::kRsub});
  for (int i = 0; i < 4; ++i) {
    a.Emit({.rx = ra,
            .rx2 = ry,
            .rd = rt,
            .rq = rq,
            .rnw = 11,
            .rn8 = 11,
            .b = vp::B::kX2,
            .wr = vp::Write::kFbuf});
    a.Emit({.rx = rt,
            .rd = ru,
            .rq = rq,
            .rnw = 11,
            .rn8 = 11,
            .rpb = r_pb,
            .rimm = r_none,
            .pa = vp::Pbuf::kRowVec,
            .c0 = vp::C0::kPbuf,
            .b = vp::B::kImm,
            .wr = vp::Write::kFbuf});
    a.Emit({.rx = ry,
            .rx2 = ru,
            .rd = ry,
            .rq = rq,
            .rnw = 11,
            .rn8 = 11,
            .b = vp::B::kX2,
            .wr = vp::Write::kFbuf});
  }
}

Pass LaneCmp(vp::Cmp cmp, vp::Sel sel) {
  return {
      .cmp = cmp, .cmp_src = vp::CmpSrc::kLane, .cmp_int = true, .sel = sel};
}

void FrameStartTime(vp::Assembler& a) {
  for (const auto& [rd, off, col] :
       {std::array{12U, kFsTs, 0U}, std::array{13U, kFsTs, 1U},
        std::array{14U, kFsPts, 0U}, std::array{15U, kFsPts, 1U}}) {
    a.Memory(vp::Smem::kSld, {.rd = rd, .ra = 1, .col = col, .imm = off});
  }
  a.Alu(vp::Salu::kSub, 16, 12, 14);
  a.Alu(vp::Salu::kSltu, 17, 12, 14);
  a.Alu(vp::Salu::kSub, 18, 13, 15);
  a.Alu(vp::Salu::kSub, 18, 18, 17);
  a.Alu(vp::Salu::kSra, 19, 16, 0, 31);
  a.Alu(vp::Salu::kSub, 20, 18, 19);
  a.Alu(vp::Salu::kSra, 21, 16, 0, 16);
  a.Alu(vp::Salu::kAnd, 22, 16, 0, 0xFFFF);
  a.Alu(vp::Salu::kXor, 23, 16, 19);
  a.Alu(vp::Salu::kSub, 23, 23, 19);
  a.Alu(vp::Salu::kSltu, 23, 23, 0, kFsMaxMicroseconds + 1);
  a.Alu(vp::Salu::kSeq, 24, 20, 0);
  a.Alu(vp::Salu::kAnd, 9, 23, 24);
  a.Alu(vp::Salu::kAnd, 9, 9, 8);
  a.Alu(vp::Salu::kSeq, 7, 16, 0);
  a.Alu(vp::Salu::kXor, 7, 7, 0, 1);
  a.Alu(vp::Salu::kAnd, 7, 7, 9);
  a.Alu(vp::Salu::kAdd, 25, 1, 0, kFsW);
  a.Alu(vp::Salu::kAdd, 26, 1, 0, kFsCol8);
  a.Alu(vp::Salu::kAdd, 27, 1, 0, kFsW + 1);
  a.LoadImmediate(28, 2);
  Pass zero = LaneCmp(vp::Cmp::kLt, vp::Sel::kZero);
  zero.rx = 25;
  zero.rd = 25;
  zero.rq = 11;
  zero.rnw = 11;
  zero.rn8 = 11;
  zero.wr = vp::Write::kFbuf;
  a.Emit(zero);
  a.StoreScalar(25, 26, 0, 20, 11);
  a.StoreScalar(25, 26, 11, 21, 11);
  a.StoreScalar(25, 26, 28, 22, 11);
  a.LoadImmediate(28, kI2f0);
  a.LoadImmediate(29, FloatBits(-kCf));
  a.IntToFloat(25, 25, 11, 11, 11, 28, 29, 27);
}

}  // namespace

vp::Assembler FrameStartKernel(uint32_t nkp, uint32_t ns) {
  vp::Assembler a(true);
  constexpr uint32_t kW0 = kFsW;
  constexpr uint32_t kW1 = kFsW + 1;
  constexpr uint32_t kDtr = kFsW + 2;
  constexpr uint32_t kDu = kFsW + 3;
  constexpr uint32_t kAbs = kFsW + 4;
  constexpr uint32_t kSgn = kFsW + 5;
  constexpr uint32_t kRc = kFsW + 6;
  constexpr uint32_t kWt = kFsW + 7;
  constexpr uint32_t kWu = kFsW + 8;
  constexpr uint32_t kInv = kFsW + 9;
  constexpr uint32_t kMw1 = kFsW + 10;
  constexpr uint32_t kMw = kFsW + 11;
  constexpr uint32_t kTc = kFsW + 12;
  constexpr uint32_t kC1 = kFsW + 13;
  constexpr uint32_t kC2 = kFsW + 14;
  constexpr uint32_t kC3 = kFsW + 15;
  constexpr uint32_t kKw = kFsW + 16;
  constexpr uint32_t kKr = kKw + 16;
  constexpr uint32_t kKt = kKw + 32;
  constexpr uint32_t kKu = kKw + 48;
  constexpr uint32_t kKx = kKw + 64;
  const uint32_t one = FloatBits(1);
  a.LoadImmediate(11, 1);
  FrameStartTime(a);
  FrameStart f(a);
  f.P({.rx = kW0, .rd = kW1, .imm = FloatBits(65536)},
      {.b = vp::B::kImm, .px = vp::Perm::kBcast, .pcol = 1});
  f.P({.rx = kW0, .rd = kW1, .rx2 = kW1},
      {.c1 = vp::C1::kX2, .px = vp::Perm::kBcast, .pcol = 2});
  f.P({.rx = kW0, .rd = kW1, .rx2 = kW1, .imm = FloatBits(int64_t{1} << 32)},
      {.b = vp::B::kImm, .c1 = vp::C1::kX2, .px = vp::Perm::kBcast, .pcol = 0});
  f.P({.rx = kW1, .rd = kDtr, .imm = Decimal(1, 1000000)}, {.b = vp::B::kImm});
  const uint32_t r_d = f.At(kDtr);
  const uint32_t r_u = f.At(kDu);
  const uint32_t r_h = f.K(Decimal(1, 2));
  a.Branch(vp::Cond::kZ, 7, "fs_default");
  a.Emit({.rx = r_d,
          .rd = r_u,
          .rq = 11,
          .rnw = 11,
          .rn8 = 11,
          .wr = vp::Write::kFbuf});
  a.Branch(vp::Cond::kAlways, 0, "fs_dt");
  a.Label("fs_default");
  Pass fallback = LaneCmp(vp::Cmp::kLt, vp::Sel::kImm);
  fallback.rx = r_d;
  fallback.rd = r_u;
  fallback.rq = 11;
  fallback.rnw = 11;
  fallback.rn8 = 11;
  fallback.rimm = r_h;
  fallback.wr = vp::Write::kFbuf;
  a.Emit(fallback);
  a.Label("fs_dt");
  f.P({.rx = kDu, .rd = kAbs, .imm = 0x7FFFFFFFU},
      {.c0 = vp::C0::kImm, .ialu = vp::Ialu::kAnd});
  f.P({.rx = kDu, .rd = kSgn, .imm = kSignBit},
      {.c0 = vp::C0::kImm, .ialu = vp::Ialu::kAnd});
  {
    const uint32_t ra = f.At(kAbs);
    const uint32_t ry = f.At(kRc);
    const uint32_t rt = f.At(kWt);
    const uint32_t ru = f.At(kWu);
    const uint32_t magic = f.K(kMagic);
    const uint32_t none = f.K(FloatBits(-1));
    const uint32_t pb = f.Pb(kFsNtwoPbuf);
    Reciprocal(a, {ra, ry, rt, ru}, 11, magic, none, pb);
  }
  f.P({.rx = kRc, .rd = kInv, .rx2 = kSgn},
      {.c0 = vp::C0::kX2, .ialu = vp::Ialu::kXor});
  f.P({.rx = kInv, .rd = kMw1, .imm = one, .imm2 = 8},
      LaneCmp(vp::Cmp::kGe, vp::Sel::kImm));
  f.P({.rx = kMw1, .rd = kMw, .imm = one, .imm2 = 10},
      LaneCmp(vp::Cmp::kLe, vp::Sel::kImm));
  for (uint32_t l = 0; l < ns; ++l) {
    f.P({.rx = kFsSc + l, .rd = kFsSc + l, .rx2 = kMw}, {.b = vp::B::kX2});
  }
  for (uint32_t k = 0; k < 4; ++k) {
    const uint32_t pb = f.Pb(k);
    Pass pass{.rpb = pb,
              .pa = vp::Pbuf::kRowVec,
              .pa_at_mul = true,
              .b = vp::B::kPbuf,
              .px = vp::Perm::kBcast,
              .pcol = k};
    if (k == 0) {
      f.P({.rx = kFsPtg, .rd = kTc}, pass);
    } else {
      pass.c1 = vp::C1::kX2;
      f.P({.rx = kFsPtg, .rd = kTc, .rx2 = kTc}, pass);
    }
  }
  f.P({.rx = kTc, .rd = kC1}, {.px = vp::Perm::kTswap1});
  f.P({.rx = kTc, .rd = kC2}, {.px = vp::Perm::kTswap2});
  f.P({.rx = kC2, .rd = kC3}, {.px = vp::Perm::kTswap1});
  const auto below = [&](uint32_t src, uint32_t dst, uint32_t lt) {
    f.P({.rx = src, .rd = dst, .rx2 = dst, .imm2 = lt},
        LaneCmp(vp::Cmp::kLt, vp::Sel::kX2));
  };
  const auto lane = [&](uint32_t src, uint32_t dst, uint32_t to, uint32_t col) {
    Pass pass = LaneCmp(vp::Cmp::kEq, vp::Sel::kX2);
    pass.px = vp::Perm::kBcast;
    pass.pcol = col;
    f.P({.rx = src, .rd = dst, .rx2 = dst, .imm2 = to}, pass);
  };
  for (const auto& [word, src] : {std::pair{0U, kTc},
                                  {1U, kC1},
                                  {2U, kC2},
                                  {8U, kC3},
                                  {9U, kTc},
                                  {10U, kC1},
                                  {11U, kC2}}) {
    below(src, kFsDi + word, 3);
  }
  for (const auto& [word, src] : {std::pair{6U, kC1}, {7U, kTc}}) {
    lane(src, kFsDi + word, 6, 1);
    lane(src, kFsDi + word, 7, 0);
  }
  for (uint32_t p = 0; p < 4; ++p) {
    for (const auto& [word, src] :
         {std::pair{2 * p, kTc}, {(2 * p) + 1, kC1}, {8U, kC3}}) {
      lane(src, kFsMi + word, 2 * p, 0);
      lane(src, kFsMi + word, (2 * p) + 1, 1);
    }
  }
  f.P({.rx = kDtr, .rd = kFsDt}, LaneCmp(vp::Cmp::kEq, vp::Sel::kZero));
  f.P({.rx = kFsKp, .rd = kKw, .rq = nkp, .imm = one, .imm2 = kConstRwx},
      LaneCmp(vp::Cmp::kGe, vp::Sel::kImm));
  f.P({.rx = kKw, .rd = kKw, .rq = nkp, .imm = one, .imm2 = kConstRwy},
      LaneCmp(vp::Cmp::kLe, vp::Sel::kImm));
  {
    const uint32_t ra = f.At(kKw);
    const uint32_t ry = f.At(kKr);
    const uint32_t rt = f.At(kKt);
    const uint32_t ru = f.At(kKu);
    const uint32_t rq = f.K(nkp);
    const uint32_t magic = f.K(kMagic);
    const uint32_t none = f.K(FloatBits(-1));
    const uint32_t pb = f.Pb(kFsNtwoPbuf);
    Reciprocal(a, {ra, ry, rt, ru}, rq, magic, none, pb);
  }
  f.P({.rx = kFsKp, .rd = kKx, .rx2 = kKr, .rq = nkp, .imm2 = kConstRwx},
      LaneCmp(vp::Cmp::kLt, vp::Sel::kX2));
  f.P({.rx = kFsKp, .rd = kFsKp, .rx2 = kKx, .rq = nkp, .imm2 = kConstRwy},
      LaneCmp(vp::Cmp::kGt, vp::Sel::kX2));
  f.P({.rx = kFsFlag, .rd = kFsFlag}, LaneCmp(vp::Cmp::kLt, vp::Sel::kZero));
  {
    const uint32_t flag = f.At(kFsFlag);
    const uint32_t col8 = f.At(kFsCol8);
    a.StoreScalar(flag, col8, 0, 9, 11);
  }
  f.P({.rx = kFsTs, .rd = kFsPts}, {});
  f.P({.rx = kFsTg, .rd = kFsPtg}, {});
  a.Ret();
  return a;
}

vp::Assembler MaskKernel(bool fill) {
  vp::Assembler a(true);
  a.LoadImmediate(11, 1);
  a.Memory(vp::Smem::kSld, {.rd = 12, .ra = 2, .col = 0});
  a.Branch(vp::Cond::kNz, 12, "done");
  if (fill) {
    a.Emit({.rx = 1,
            .rd = 3,
            .rq = 4,
            .rnw = 11,
            .rn8 = 11,
            .rimm = 8,
            .wr = vp::Write::kFbuf,
            .cmp = vp::Cmp::kLt,
            .cmp_src = vp::CmpSrc::kLane,
            .cmp_int = true,
            .sel = vp::Sel::kImm});
  } else {
    a.Emit({.rx = 1,
            .rd = 3,
            .rq = 4,
            .rnw = 11,
            .rn8 = 11,
            .wr = vp::Write::kFbuf});
  }
  a.Label("done");
  a.Ret();
  return a;
}

namespace {

class Queue {
 public:
  explicit Queue(vp::Assembler& a) : a_(a) {
    a_.LoadImmediate(11, 1);
    a_.LoadImmediate(12, 4);
    a_.Alu(vp::Salu::kShl, 13, 4, 0, 2);
    a_.Alu(vp::Salu::kAdd, 14, 2, 13);
    a_.Alu(vp::Salu::kAdd, 15, 1, 0, 1);
    a_.Alu(vp::Salu::kAdd, 16, 1, 0, 2);
    a_.Alu(vp::Salu::kAdd, 17, 1, 0, 3);
    a_.Alu(vp::Salu::kAdd, 18, 2, 0, 2);
    a_.Alu(vp::Salu::kAdd, 19, 2, 0, 3);
    a_.Alu(vp::Salu::kAdd, 20, 14, 0, 1);
    a_.Alu(vp::Salu::kAdd, 21, 14, 0, 2);
    a_.Alu(vp::Salu::kAdd, 22, 14, 0, 3);
  }

  void P(Pass pass) {
    pass.rq = 4;
    pass.rnw = 11;
    pass.rn8 = 12;
    pass.wr = vp::Write::kFbuf;
    a_.Emit(pass);
  }

  void Lane(uint32_t rx, uint32_t rd, std::optional<uint32_t> rx2, vp::Cmp cmp,
            vp::Sel sel, uint32_t rimm, bool dst_s1) {
    P({.rx = rx,
       .rx2 = rx2.value_or(0),
       .rd = rd,
       .rimm = rimm,
       .rimm2 = 28,
       .cmp = cmp,
       .cmp_src = vp::CmpSrc::kLane,
       .cmp_int = true,
       .sel = sel,
       .dst_s1 = dst_s1});
  }

 private:
  vp::Assembler& a_;
};

}  // namespace

vp::Assembler IqueueState(uint32_t ncls) {
  vp::Assembler a(true);
  Queue q(a);
  for (const auto& [reg, col] : {std::pair{23U, kQkOne},
                                 {24U, kQkFna},
                                 {25U, kQkNeg},
                                 {26U, kQkFtq},
                                 {27U, kQkLthr}}) {
    a.Memory(vp::Smem::kSld, {.rd = reg, .ra = 8, .col = col});
  }
  q.P({.rx = 14, .rd = 20, .px = vp::Perm::kBcast, .pcol = 2});
  q.P({.rx = 1, .rd = 15, .px = vp::Perm::kBcast, .pcol = 1});
  q.P({.rx = 1, .rd = 16, .px = vp::Perm::kBcast, .pcol = 0});
  q.P({.rx = 1, .rd = 17, .px = vp::Perm::kBcast, .pcol = 2});
  q.P({.rx = 16, .rx2 = 15, .rd = 16, .b = vp::B::kX2});
  q.P({.rx = 16, .rd = 16, .rimm = 23, .c0 = vp::C0::kImm});
  q.P({.rx = 16,
       .rd = 16,
       .rimm = 26,
       .rimm2 = 26,
       .cmp = vp::Cmp::kLe,
       .cmp_src = vp::CmpSrc::kImm2,
       .sel = vp::Sel::kImm});
  q.P({.rx = 16,
       .rx2 = 20,
       .rd = 16,
       .rimm2 = 24,
       .cmp = vp::Cmp::kLe,
       .cmp_src = vp::CmpSrc::kX2Imm2,
       .sel = vp::Sel::kZero});
  a.LoadImmediate(28, (ncls + 7) / 8);
  a.LoadImmediate(29, 0);
  a.Move(30, 2);
  a.Move(31, 18);
  a.Label("rm");
  a.Emit({.rx = 30,
          .rq = 11,
          .rnw = 28,
          .rn8 = 12,
          .rimm2 = 5,
          .mask_en = true,
          .red = vp::Reduce::kMax,
          .sc = vp::RowScalar::kMax});
  a.Emit({.rx = 31,
          .rd = 31,
          .rq = 11,
          .rnw = 11,
          .rn8 = 11,
          .rimm2 = 0,
          .wr = vp::Write::kFbuf,
          .cmp = vp::Cmp::kLt,
          .cmp_src = vp::CmpSrc::kLane,
          .sel = vp::Sel::kRs1});
  a.Alu(vp::Salu::kAdd, 30, 30, 12);
  a.Alu(vp::Salu::kAdd, 31, 31, 12);
  a.Alu(vp::Salu::kAdd, 29, 29, 0, 1);
  a.Alu(vp::Salu::kSub, 13, 4, 29);
  a.Branch(vp::Cond::kNz, 13, "rm");
  q.P({.rx = 18,
       .rd = 18,
       .rimm = 23,
       .rimm2 = 27,
       .c0 = vp::C0::kZero,
       .cmp = vp::Cmp::kLe,
       .cmp_src = vp::CmpSrc::kImm2,
       .sel = vp::Sel::kImm,
       .ialu = vp::Ialu::kAnd});
  q.P({.rx = 18,
       .rx2 = 20,
       .rd = 18,
       .rimm2 = 24,
       .cmp = vp::Cmp::kLt,
       .cmp_src = vp::CmpSrc::kX2Imm2,
       .sel = vp::Sel::kZero});
  q.P({.rx = 18,
       .rx2 = 20,
       .rd = 18,
       .rimm = 23,
       .rimm2 = 24,
       .cmp = vp::Cmp::kNe,
       .cmp_src = vp::CmpSrc::kX2Imm2,
       .sel = vp::Sel::kImm});
  q.P({.rx = 16,
       .rd = 1,
       .rimm2 = 0,
       .cmp = vp::Cmp::kEq,
       .cmp_src = vp::CmpSrc::kLane,
       .sel = vp::Sel::kZero});
  a.LoadImmediate(28, 1);
  q.P({.rx = 18,
       .rx2 = 1,
       .rd = 1,
       .rimm2 = 28,
       .cmp = vp::Cmp::kEq,
       .cmp_src = vp::CmpSrc::kLane,
       .sel = vp::Sel::kX2});
  a.LoadImmediate(28, 2);
  q.P({.rx = 20,
       .rx2 = 1,
       .rd = 1,
       .rimm2 = 28,
       .cmp = vp::Cmp::kEq,
       .cmp_src = vp::CmpSrc::kLane,
       .sel = vp::Sel::kX2});
  q.P({.rx = 16,
       .rd = 21,
       .rpb = 6,
       .pa = vp::Pbuf::kRowVec,
       .c0 = vp::C0::kPbuf});
  q.P({.rx = 21,
       .rd = 21,
       .rimm = 25,
       .rimm2 = 26,
       .c0 = vp::C0::kZero,
       .cmp = vp::Cmp::kGe,
       .cmp_src = vp::CmpSrc::kImm2,
       .sel = vp::Sel::kImm,
       .ialu = vp::Ialu::kAnd});
  a.LoadImmediate(28, 4);
  q.Lane(21, 22, std::nullopt, vp::Cmp::kLt, vp::Sel::kImm, 25, false);
  a.LoadImmediate(28, 12);
  q.Lane(21, 22, 22, vp::Cmp::kGe, vp::Sel::kX2, 0, false);
  a.LoadImmediate(28, 16);
  q.Lane(22, 3, std::nullopt, vp::Cmp::kLt, vp::Sel::kImm, 25, true);
  a.Alu(vp::Salu::kAdd, 13, 3, 4);
  a.LoadImmediate(28, 20);
  q.Lane(21, 22, std::nullopt, vp::Cmp::kLt, vp::Sel::kImm, 25, false);
  a.LoadImmediate(28, 16);
  q.Lane(22, 22, std::nullopt, vp::Cmp::kGe, vp::Sel::kImm, 25, false);
  a.LoadImmediate(28, 28);
  q.Lane(21, 13, 22, vp::Cmp::kGe, vp::Sel::kX2, 0, true);
  a.Ret();
  return a;
}

vp::Assembler IqueueList() {
  vp::Assembler a(true);
  Queue q(a);
  for (const auto& [reg, col] : {std::pair{23U, kQlTb4},
                                 {24U, kQlBaseO},
                                 {25U, kQlTb1},
                                 {26U, kQlTb1 + 1},
                                 {27U, kQlTb1 + 2},
                                 {29U, kQlI2f0},
                                 {30U, kQlCf}}) {
    a.Memory(vp::Smem::kSld, {.rd = reg, .ra = 8, .col = col});
  }
  q.P({.rx = 17, .rd = 21, .rimm = 23, .b = vp::B::kImm});
  a.RoundToInt(21, 21, 4, 11, 12, 29, 30, 22);
  q.P({.rx = 21,
       .rd = 21,
       .rimm = 24,
       .c0 = vp::C0::kImm,
       .ialu = vp::Ialu::kAdd});
  constexpr std::array<uint32_t, 6> kColumns = {0, 1, 2, 4, 5, 6};
  for (size_t k = 0; k < kColumns.size(); ++k) {
    const uint32_t c = kColumns[k];
    a.LoadImmediate(28, c);
    q.P({.rx = 21,
         .rx2 = k == 0 ? 0U : 19U,
         .rd = 19,
         .rimm = 25 + (c % 4),
         .rimm2 = 28,
         .c0 = vp::C0::kImm,
         .cmp = vp::Cmp::kEq,
         .cmp_src = vp::CmpSrc::kLane,
         .sel = k == 0 ? vp::Sel::kZero : vp::Sel::kX2,
         .ialu = vp::Ialu::kAdd});
  }
  q.P({.rx = 19,
       .rx2 = 15,
       .rd = 19,
       .rimm2 = 0,
       .cmp = vp::Cmp::kGt,
       .cmp_src = vp::CmpSrc::kX2Imm2,
       .sel = vp::Sel::kNull});
  a.Memory(vp::Smem::kSld, {.rd = 23, .ra = 8, .col = kQlTbC});
  a.Memory(vp::Smem::kSld, {.rd = 24, .ra = 8, .col = kQlBaseC});
  a.Memory(vp::Smem::kSld, {.rd = 25, .ra = 8, .col = kQlFna});
  q.P({.rx = 20, .rd = 21, .rimm = 23, .b = vp::B::kImm});
  a.RoundToInt(21, 21, 4, 11, 12, 29, 30, 22);
  q.P({.rx = 21,
       .rd = 21,
       .rimm = 24,
       .c0 = vp::C0::kImm,
       .ialu = vp::Ialu::kAdd});
  q.P({.rx = 21,
       .rx2 = 20,
       .rd = 21,
       .rimm2 = 25,
       .cmp = vp::Cmp::kLe,
       .cmp_src = vp::CmpSrc::kX2Imm2,
       .sel = vp::Sel::kNull});
  for (const uint32_t lane : {3U, 7U}) {
    a.LoadImmediate(28, lane);
    q.P({.rx = 21,
         .rx2 = 19,
         .rd = 19,
         .rimm2 = 28,
         .cmp = vp::Cmp::kEq,
         .cmp_src = vp::CmpSrc::kLane,
         .sel = vp::Sel::kX2});
  }
  a.Alu(vp::Salu::kShr, 29, 4, 0, 1);
  a.LoadImmediate(30, 8);
  a.Alu(vp::Salu::kAdd, 13, 19, 0, 4);
  a.LoadImmediate(28, 4);
  a.Emit({.rx = 19,
          .rx2 = 13,
          .rd = 3,
          .rq = 29,
          .rnw = 11,
          .rn8 = 30,
          .rimm2 = 28,
          .wr = vp::Write::kFbuf,
          .cmp = vp::Cmp::kLt,
          .cmp_src = vp::CmpSrc::kLane,
          .sel = vp::Sel::kX2,
          .dst_s1 = true});
  a.Emit({.rx = 3,
          .rd = 3,
          .rq = 29,
          .rnw = 11,
          .rn8 = 11,
          .wr = vp::Write::kFbuf,
          .px = vp::Perm::kT4b});
  a.Ret();
  return a;
}

vp::Assembler IqueueStlist(uint32_t na) {
  vp::Assembler a(true);
  a.LoadImmediate(11, 1);
  a.LoadImmediate(12, 4);
  for (const auto& [reg, col] : {std::pair{23U, kQsFnt},
                                 {24U, kQsTb},
                                 {25U, kQsBase},
                                 {26U, kQsENa},
                                 {29U, kQsI2f0},
                                 {30U, kQsCf}}) {
    a.Memory(vp::Smem::kSld, {.rd = reg, .ra = 8, .col = col});
  }
  const auto p = [&](Pass pass) {
    pass.rq = 4;
    pass.rnw = 11;
    pass.rn8 = 11;
    pass.wr = vp::Write::kFbuf;
    a.Emit(pass);
  };
  a.Alu(vp::Salu::kAdd, 13, 3, 4);
  p({.rx = 1, .rd = 3, .rimm = 24, .b = vp::B::kImm});
  a.RoundToInt(3, 3, 4, 11, 11, 29, 30, 13);
  p({.rx = 3, .rd = 3, .rimm = 25, .c0 = vp::C0::kImm, .ialu = vp::Ialu::kAdd});
  p({.rx = 3,
     .rx2 = 2,
     .rd = 3,
     .rimm2 = 23,
     .cmp = vp::Cmp::kLt,
     .cmp_src = vp::CmpSrc::kX2Imm2,
     .sel = vp::Sel::kNull});
  a.Alu(vp::Salu::kAdd, 14, 3, 0, na / 32);
  a.LoadImmediate(15, na % 32);
  a.Emit({.rx = 14,
          .rd = 14,
          .rq = 11,
          .rnw = 11,
          .rn8 = 11,
          .rimm = 26,
          .rimm2 = 15,
          .wr = vp::Write::kFbuf,
          .cmp = vp::Cmp::kNe,
          .cmp_src = vp::CmpSrc::kLane,
          .cmp_int = true,
          .sel = vp::Sel::kImm});
  a.Ret();
  return a;
}

vp::Assembler IqueueEgo() {
  vp::Assembler a(true);
  a.LoadImmediate(11, 1);
  a.LoadImmediate(15, 1);
  a.Memory(vp::Smem::kSld, {.rd = 14, .ra = 2, .col = 6});
  a.Emit({.rx = 3,
          .rd = 3,
          .rq = 11,
          .rnw = 11,
          .rn8 = 11,
          .rimm = 14,
          .rimm2 = 15,
          .wr = vp::Write::kFbuf,
          .cmp = vp::Cmp::kNe,
          .cmp_src = vp::CmpSrc::kLane,
          .cmp_int = true,
          .sel = vp::Sel::kImm});
  a.Ret();
  return a;
}

uint32_t CbufImage::Add(const std::string& name,
                        const vprog::Assembler& program) {
  const auto pc = static_cast<uint32_t>(vprog::kRomWords + words_.size());
  const std::vector<Instruction> words = program.Words(pc);
  words_.insert(words_.end(), words.begin(), words.end());
  pcs_[name] = pc;
  return pc;
}

}  // namespace hotarubi::compiler::sparsedrive
