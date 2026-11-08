#ifndef HOTARUBI_TOOLS_COMPILER_VPROG_HPP_
#define HOTARUBI_TOOLS_COMPILER_VPROG_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "isa.hpp"

namespace hotarubi::compiler::vprog {

inline constexpr uint32_t kCbufWords = 1024;
inline constexpr uint32_t kRomWords = 256;
inline constexpr uint32_t kNull = 0xFFFFFFFFU;

enum class TransposePack : uint8_t { kNone = 0, kZip4 = 1 };
enum class WordMap : uint8_t { kNorm = 0, kX1 = 1, kX2 = 2, kS4 = 3 };
enum class Pbuf : uint8_t { kNone = 0, kRowVec = 1, kTile = 2 };
enum class C0 : uint8_t { kZero = 0, kX2 = 1, kPbuf = 2, kRs0 = 3, kImm = 4 };
enum class B : uint8_t {
  kOne = 0,
  kX2 = 1,
  kPbuf = 2,
  kRs1 = 3,
  kImm = 4,
  kPre = 5,
  kLog2e = 6
};
enum class Nl : uint8_t { kNone = 0, kAct = 1, kExp2 = 2, kTab01 = 3 };
enum class C1 : uint8_t { kZero = 0, kX2 = 1, kMax = 2, kSignfix = 3 };
enum class Reduce : uint8_t { kNone = 0, kSum = 1, kMax = 2 };
enum class Write : uint8_t { kNone = 0, kFbuf = 1, kAbuf = 2, kFinal = 3 };
enum class RowScalar : uint8_t {
  kNone = 0,
  kMean = 1,
  kRstd = 2,
  kNegMax = 3,
  kRecipSum = 4,
  kSum = 5,
  kMax = 6
};
enum class Perm : uint8_t {
  kNone = 0,
  kBcast = 1,
  kTswap1 = 2,
  kTswap2 = 3,
  kT4 = 4,
  kT4b = 5
};
enum class Cmp : uint8_t {
  kNone = 0,
  kLt = 1,
  kLe = 2,
  kEq = 3,
  kNe = 4,
  kGt = 5,
  kGe = 6
};
enum class CmpSrc : uint8_t { kImm2 = 0, kX2 = 1, kLane = 2, kX2Imm2 = 3 };
enum class Sel : uint8_t {
  kNone = 0,
  kZero = 1,
  kX2 = 2,
  kImm = 3,
  kNull = 4,
  kRs0 = 5,
  kRs1 = 6
};
enum class Ialu : uint8_t {
  kNone = 0,
  kAdd = 1,
  kSub = 2,
  kAnd = 3,
  kOr = 4,
  kXor = 5,
  kShr1 = 6,
  kRsub = 7
};
enum class Salu : uint8_t {
  kAdd = 0,
  kSub = 1,
  kAnd = 2,
  kOr = 3,
  kXor = 4,
  kShl = 5,
  kShr = 6,
  kSra = 7,
  kMul = 8,
  kSlt = 9,
  kSltu = 10,
  kSeq = 11,
  kMov = 12
};
enum class Smem : uint8_t {
  kSld = 0,
  kRsld = 2,
  kRsmov = 4,
  kSmovr = 5,
  kRn = 6
};
enum class Cond : uint8_t { kAlways = 0, kZ = 1, kNz = 2 };

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
  WordMap wx = WordMap::kNorm;
  WordMap wx2 = WordMap::kNorm;
  WordMap wd = WordMap::kNorm;
  Pbuf pa = Pbuf::kNone;
  bool pa_at_mul = false;
  C0 c0 = C0::kZero;
  B b = B::kOne;
  Nl nl = Nl::kNone;
  C1 c1 = C1::kZero;
  bool neg_c1 = false;
  bool mask_en = false;
  Reduce red = Reduce::kNone;
  Write wr = Write::kNone;
  RowScalar sc = RowScalar::kNone;
  Perm px = Perm::kNone;
  uint32_t pcol = 0;
  Cmp cmp = Cmp::kNone;
  CmpSrc cmp_src = CmpSrc::kImm2;
  bool cmp_int = false;
  Sel sel = Sel::kNone;
  Ialu ialu = Ialu::kNone;
  bool dst_s1 = false;
  bool c0_pred = false;
  bool nodrain = false;
  bool x_s1 = false;
  uint32_t ctx = 0;
};

struct SmemArgs {
  uint32_t rd = 0;
  uint32_t ra = 0;
  uint32_t rb = 0;
  std::optional<uint32_t> col = std::nullopt;
  bool which = false;
  bool neg = false;
  uint32_t ctx = 0;
  uint32_t imm = 0;
};

Instruction EncodePass(const Pass& pass);
Instruction EncodeSalu(Salu op, uint32_t rd, uint32_t ra, uint32_t rb,
                       std::optional<uint32_t> imm);
Instruction EncodeSmem(Smem op, const SmemArgs& args);
Instruction EncodeBranch(Cond cond, uint32_t ra, uint32_t target);
Instruction EncodeRet();
Instruction EncodeNop();

class Assembler {
 public:
  explicit Assembler(bool raw_copy = false) : raw_copy_(raw_copy) {}

  [[nodiscard]] size_t Size() const { return size_; }
  void Label(std::string_view name);
  Assembler& Emit(const Pass& pass);
  Assembler& Alu(Salu op, uint32_t rd, uint32_t ra, uint32_t rb = 0,
                 std::optional<uint32_t> imm = std::nullopt);
  Assembler& LoadImmediate(uint32_t rd, uint32_t imm);
  Assembler& Move(uint32_t rd, uint32_t rb);
  Assembler& Memory(Smem op, const SmemArgs& args);
  Assembler& Branch(Cond cond, uint32_t ra, std::string_view label);
  Assembler& Ret();
  Assembler& Nop();
  Assembler& Transpose8(uint32_t src, uint32_t dst, uint32_t tiles,
                        uint32_t src_tile_stride, uint32_t src_word_stride,
                        uint32_t dst_tile_stride, uint32_t dst_word_stride,
                        TransposePack pack = TransposePack::kNone);

  Assembler& StoreScalar(uint32_t rw, uint32_t rlane, uint32_t rcol,
                         uint32_t rval, uint32_t r_one);
  Assembler& StoreRowScalar(uint32_t rw, uint32_t rcol8, uint32_t rcol,
                            bool rs0, uint32_t r_one);
  Assembler& IntToFloat(uint32_t rx, uint32_t rd, uint32_t rq, uint32_t rnw,
                        uint32_t rn8, uint32_t r_c0, uint32_t r_ncf,
                        uint32_t rtmp);
  Assembler& RoundToInt(uint32_t rx, uint32_t rd, uint32_t rq, uint32_t rnw,
                        uint32_t rn8, uint32_t r_c0, uint32_t r_cf,
                        uint32_t rtmp);
  Assembler& RoundToIntFloat(uint32_t rx, uint32_t rd, uint32_t rq,
                             uint32_t rnw, uint32_t rn8, uint32_t r_cf,
                             uint32_t r_ncf, uint32_t rtmp);
  Assembler& FloorFloat(uint32_t rx, uint32_t rd, uint32_t rq, uint32_t rnw,
                        uint32_t rn8, uint32_t r_cf, uint32_t r_ncf,
                        uint32_t r_none, uint32_t rtmp);

  [[nodiscard]] std::vector<Instruction> Words(uint32_t base = 0) const;

 private:
  struct BranchItem {
    Cond cond;
    uint32_t ra;
    std::string label;
  };
  using Item = std::variant<Instruction, Pass, BranchItem>;

  bool raw_copy_;
  std::vector<Item> items_;
  std::map<std::string, size_t, std::less<>> labels_;
  size_t size_ = 0;
};

}  // namespace hotarubi::compiler::vprog

#endif  // HOTARUBI_TOOLS_COMPILER_VPROG_HPP_
