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

enum class Map : uint8_t { kNone = 0, kGroup = 1, kUpper = 2 };

struct WordMap {
  uint32_t group = 0;
  uint32_t stride = 0;
  Map x = Map::kNone;
  Map x2 = Map::kNone;
  Map dst = Map::kNone;
};

inline constexpr uint32_t kLog2e = 0x3FB8AA3B;

enum class Pbuf : uint8_t { kNone = 0, kRowVec = 1, kTile = 2 };
enum class C0 : uint8_t { kZero = 0, kX2 = 1, kPbuf = 2, kRs0 = 3, kImm = 4 };
enum class B : uint8_t {
  kOne = 0,
  kX2 = 1,
  kPbuf = 2,
  kRs1 = 3,
  kImm = 4,
  kPre = 5
};
enum class Nl : uint8_t { kNone = 0, kAct = 1, kExp2 = 2, kTab01 = 3 };
enum class Post : uint8_t { kAdd = 0, kSub = 1, kMax = 2, kSignfix = 3 };
enum class Reduce : uint8_t { kNone = 0, kSum = 1, kMax = 2 };
enum class Write : uint8_t { kNone = 0, kFbuf = 1, kAbuf = 2, kFinal = 3 };
enum class ScalarFn : uint8_t {
  kNone = 0,
  kIdentity = 1,
  kNegate = 2,
  kReciprocal = 3,
  kRsqrt = 4
};

struct RowScalar {
  ScalarFn fn = ScalarFn::kNone;
  bool rs0 = false;
  bool rn = false;
  bool eps = false;
};

inline constexpr RowScalar kMean{
    .fn = ScalarFn::kNegate, .rs0 = true, .rn = true, .eps = false};
inline constexpr RowScalar kRstd{
    .fn = ScalarFn::kRsqrt, .rs0 = false, .rn = true, .eps = true};
inline constexpr RowScalar kNegMax{
    .fn = ScalarFn::kNegate, .rs0 = true, .rn = false, .eps = false};
inline constexpr RowScalar kRecipSum{
    .fn = ScalarFn::kReciprocal, .rs0 = false, .rn = false, .eps = false};
inline constexpr RowScalar kMax{
    .fn = ScalarFn::kIdentity, .rs0 = false, .rn = false, .eps = false};

inline constexpr uint32_t kPermSame = 0;
inline constexpr uint32_t kPermZero = 6;
inline constexpr uint32_t kPermOne = 7;
inline constexpr uint32_t kPermInvert = 8;

constexpr uint32_t PermBit(uint32_t bit, uint32_t selector) {
  return selector << (4 * bit);
}

constexpr uint32_t PermPosition(uint32_t position_bit) {
  return 5 - position_bit;
}

constexpr uint32_t Bcast(uint32_t column) {
  uint32_t code = 0;
  for (uint32_t k = 0; k < 3; ++k) {
    code |= PermBit(k, ((column >> k) & 1U) != 0 ? kPermOne : kPermZero);
  }
  return code;
}

constexpr uint32_t TokenXor(uint32_t mask) {
  return ((mask & 1U) != 0 ? PermBit(3, kPermSame | kPermInvert) : 0) |
         ((mask & 2U) != 0 ? PermBit(4, kPermSame | kPermInvert) : 0);
}

inline constexpr uint32_t kTokensToLanes =
    PermBit(4, PermPosition(1)) | PermBit(3, PermPosition(0)) |
    PermBit(2, PermPosition(4)) | PermBit(1, PermPosition(3)) |
    PermBit(0, PermPosition(2));
inline constexpr uint32_t kTokenPairsToLanes = PermBit(4, PermPosition(3)) |
                                               PermBit(3, PermPosition(2)) |
                                               PermBit(2, PermPosition(4));

enum class Cmp : uint8_t {
  kNone = 0,
  kLt = 1,
  kLe = 2,
  kEq = 3,
  kNe = 4,
  kGt = 5,
  kGe = 6
};
enum class CmpLeft : uint8_t { kX = 0, kX2 = 1, kColumn = 2, kPosition = 3 };
enum class Sel : uint8_t {
  kNone = 0,
  kZero = 1,
  kX2 = 2,
  kImm = 3,
  kNull = 4,
  kRs1 = 5
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
enum class RowMask : uint8_t { kNone = 0, kSkip = 1, kSet = 2 };
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
enum class Smem : uint8_t { kSld = 0, kRsmov = 1, kRn = 2 };
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
  WordMap map{};
  Pbuf pa = Pbuf::kNone;
  C0 c0 = C0::kZero;
  B b = B::kOne;
  Nl nl = Nl::kNone;
  Post post = Post::kAdd;
  bool post_x2 = false;
  bool mask_en = false;
  Reduce red = Reduce::kNone;
  Write wr = Write::kNone;
  RowScalar sc{};
  uint32_t px = 0;
  Cmp cmp = Cmp::kNone;
  CmpLeft cmp_left = CmpLeft::kX;
  bool cmp_right_x2 = false;
  bool cmp_int = false;
  Sel sel = Sel::kNone;
  Ialu ialu = Ialu::kNone;
  bool dst_s1 = false;
  bool c0_pred = false;
  bool x_s1 = false;
  uint32_t ctx = 0;
  RowMask rm = RowMask::kNone;
};

struct SmemArgs {
  uint32_t rd = 0;
  uint32_t ra = 0;
  uint32_t rb = 0;
  std::optional<uint32_t> col = std::nullopt;
  bool which = false;
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
  void SkipMaskedRows(bool skip) { skip_masked_rows_ = skip; }
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
                        uint32_t rotation = 0);

  Assembler& StoreScalar(uint32_t rw, uint32_t rlane, uint32_t rcol,
                         uint32_t rval, uint32_t r_one);
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
  bool skip_masked_rows_ = false;
  std::vector<Item> items_;
  std::map<std::string, size_t, std::less<>> labels_;
  size_t size_ = 0;
};

}  // namespace hotarubi::compiler::vprog

#endif  // HOTARUBI_TOOLS_COMPILER_VPROG_HPP_
