#ifndef HOTARUBI_ISA_HPP_
#define HOTARUBI_ISA_HPP_

#include <array>
#include <cstdint>
#include <string_view>

namespace hotarubi::isa {

struct Field {
  uint32_t lsb = 0;
  uint32_t width = 0;
};

inline constexpr Field kOpcode{.lsb = 120, .width = 8};
inline constexpr Field kWait{.lsb = 116, .width = 4};

inline constexpr uint32_t kOpNop = 0;
inline constexpr uint32_t kOpDmaLoad = 1;
inline constexpr uint32_t kOpDmaStore = 2;
inline constexpr uint32_t kOpGemm = 4;
inline constexpr uint32_t kOpVec = 5;
inline constexpr uint32_t kOpSignal = 6;
inline constexpr uint32_t kOpHalt = 7;
inline constexpr uint32_t kOpWaitHost = 8;

inline constexpr uint32_t kEngineDma = 0;
inline constexpr uint32_t kEngineGemm = 1;
inline constexpr uint32_t kEngineVec = 2;
inline constexpr uint32_t kEngines = 3;

constexpr uint32_t WaitBit(uint32_t engine) { return 1U << engine; }

inline constexpr uint32_t kWaitDma = WaitBit(kEngineDma);
inline constexpr uint32_t kWaitGemm = WaitBit(kEngineGemm);
inline constexpr uint32_t kWaitVec = WaitBit(kEngineVec);
inline constexpr uint32_t kWaitAll = kWaitDma | kWaitGemm | kWaitVec;

enum class Mem : uint8_t {
  kAbuf = 0,
  kWbuf = 1,
  kFbuf = 2,
  kPbuf = 3,
  kTbuf = 4,
  kCbuf = 5,
};

inline constexpr uint32_t kNullEntry = 0xFFFFFFFFU;

namespace dma {

inline constexpr Field kDdrAddr{.lsb = 88, .width = 28};
inline constexpr Field kMem{.lsb = 85, .width = 3};
inline constexpr Field kAddr{.lsb = 72, .width = 13};
inline constexpr Field kRows{.lsb = 60, .width = 12};
inline constexpr Field kRowWords{.lsb = 48, .width = 12};
inline constexpr Field kDdrStride{.lsb = 28, .width = 20};
inline constexpr Field kKind{.lsb = 25, .width = 3};
inline constexpr Field kTok{.lsb = 19, .width = 6};
inline constexpr Field kAcc{.lsb = 18, .width = 1};
inline constexpr Field kFlush{.lsb = 17, .width = 1};

inline constexpr uint32_t kKindRaw = 0;
inline constexpr uint32_t kKindRow = 1;
inline constexpr uint32_t kKindRowIndexed = 2;
inline constexpr uint32_t kKindSplit8 = 3;
inline constexpr uint32_t kKindT32Split8 = 4;
inline constexpr uint32_t kKindGacc = 5;

inline constexpr std::array<std::string_view, 6> kKindNames = {
    "RAW", "ROW", "ROW_INDEXED", "SPLIT8", "T32_SPLIT8", "GACC"};

}  // namespace dma

namespace gemm {

inline constexpr Field kABase{.lsb = 92, .width = 20};
inline constexpr Field kAStride{.lsb = 82, .width = 10};
inline constexpr Field kKb{.lsb = 72, .width = 10};
inline constexpr Field kM{.lsb = 62, .width = 10};
inline constexpr Field kWBase{.lsb = 42, .width = 20};
inline constexpr Field kWStride{.lsb = 32, .width = 10};
inline constexpr Field kN8{.lsb = 24, .width = 8};
inline constexpr Field kCBase{.lsb = 4, .width = 20};
inline constexpr Field kWmode{.lsb = 2, .width = 2};
inline constexpr Field kAcc{.lsb = 1, .width = 1};

enum class WeightMode : uint8_t {
  kFp4 = 0,
  kW8Hi = 1,
  kW8Lo = 2,
};

}  // namespace gemm

namespace vec {

inline constexpr Field kEntry{.lsb = 108, .width = 8};
inline constexpr Field kQ{.lsb = 107, .width = 1};
inline constexpr Field kSrc{.lsb = 95, .width = 12};
inline constexpr Field kSrc2{.lsb = 83, .width = 12};
inline constexpr Field kDst{.lsb = 70, .width = 13};
inline constexpr Field kRowsQ{.lsb = 61, .width = 9};
inline constexpr Field kN8{.lsb = 53, .width = 8};
inline constexpr Field kPBase{.lsb = 43, .width = 9};
inline constexpr Field kAux{.lsb = 32, .width = 11};
inline constexpr Field kImm{.lsb = 0, .width = 32};

inline constexpr uint32_t kRegSrc = 1;
inline constexpr uint32_t kRegSrc2 = 2;
inline constexpr uint32_t kRegDst = 3;
inline constexpr uint32_t kRegRowsQ = 4;
inline constexpr uint32_t kRegN8 = 5;
inline constexpr uint32_t kRegPBase = 6;
inline constexpr uint32_t kRegAux = 7;
inline constexpr uint32_t kRegImm = 8;
inline constexpr uint32_t kRegQ = 9;

inline constexpr uint32_t kEntryWords = 4;
inline constexpr uint32_t kRomEntries = 64;
inline constexpr uint32_t kRomWords = kRomEntries * kEntryWords;

}  // namespace vec

namespace signal {

inline constexpr Field kId{.lsb = 104, .width = 8};
inline constexpr Field kValue{.lsb = 0, .width = 32};

}  // namespace signal

namespace wait_host {

inline constexpr Field kValue{.lsb = 0, .width = 32};

}  // namespace wait_host

namespace program {

inline constexpr Field kKind{.lsb = 124, .width = 4};

inline constexpr uint32_t kKindPass = 0;
inline constexpr uint32_t kKindSalu = 1;
inline constexpr uint32_t kKindSmem = 2;
inline constexpr uint32_t kKindBranch = 3;
inline constexpr uint32_t kKindRet = 4;
inline constexpr uint32_t kKindNop = 5;
inline constexpr uint32_t kKindTranspose = 7;

inline constexpr uint32_t kRegZero = 0;

inline constexpr Field kOp{.lsb = 119, .width = 5};
inline constexpr Field kRd{.lsb = 114, .width = 5};
inline constexpr Field kRa{.lsb = 109, .width = 5};
inline constexpr Field kRb{.lsb = 104, .width = 5};
inline constexpr Field kUseImm{.lsb = 103, .width = 1};
inline constexpr Field kImm{.lsb = 0, .width = 32};

inline constexpr Field kSmemCol{.lsb = 98, .width = 5};
inline constexpr Field kSmemWhich{.lsb = 97, .width = 1};
inline constexpr Field kSmemCtx{.lsb = 96, .width = 1};

inline constexpr uint32_t kSmemWhichRs1 = 0;
inline constexpr uint32_t kSmemWhichRs0 = 1;

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
  kMov = 12,
};

enum class Smem : uint8_t {
  kSld = 0,
  kRsmov = 1,
  kRn = 2,
};

enum class Cond : uint8_t {
  kAlways = 0,
  kZ = 1,
  kNz = 2,
};

}  // namespace program

namespace pass {

inline constexpr Field kRx{.lsb = 119, .width = 5};
inline constexpr Field kRx2{.lsb = 114, .width = 5};
inline constexpr Field kRd{.lsb = 109, .width = 5};
inline constexpr Field kRq{.lsb = 104, .width = 5};
inline constexpr Field kRnw{.lsb = 99, .width = 5};
inline constexpr Field kRn8{.lsb = 94, .width = 5};
inline constexpr Field kRpb{.lsb = 89, .width = 5};
inline constexpr Field kRimm{.lsb = 84, .width = 5};
inline constexpr Field kRimm2{.lsb = 79, .width = 5};
inline constexpr Field kX2En{.lsb = 78, .width = 1};
inline constexpr Field kMg{.lsb = 76, .width = 2};
inline constexpr Field kMs{.lsb = 74, .width = 2};
inline constexpr Field kMx{.lsb = 72, .width = 2};
inline constexpr Field kMx2{.lsb = 70, .width = 2};
inline constexpr Field kMd{.lsb = 68, .width = 2};
inline constexpr Field kPa{.lsb = 66, .width = 2};
inline constexpr Field kC0{.lsb = 63, .width = 3};
inline constexpr Field kB{.lsb = 60, .width = 3};
inline constexpr Field kNl{.lsb = 58, .width = 2};
inline constexpr Field kPo{.lsb = 56, .width = 2};
inline constexpr Field kPv{.lsb = 55, .width = 1};
inline constexpr Field kRed{.lsb = 53, .width = 2};
inline constexpr Field kWr{.lsb = 51, .width = 2};
inline constexpr Field kSf{.lsb = 48, .width = 3};
inline constexpr Field kSt{.lsb = 47, .width = 1};
inline constexpr Field kSrn{.lsb = 46, .width = 1};
inline constexpr Field kSeps{.lsb = 45, .width = 1};
inline constexpr Field kPx{.lsb = 25, .width = 20};
inline constexpr Field kCmp{.lsb = 22, .width = 3};
inline constexpr Field kCl{.lsb = 19, .width = 3};
inline constexpr Field kCr{.lsb = 18, .width = 1};
inline constexpr Field kCmpInt{.lsb = 17, .width = 1};
inline constexpr Field kSel{.lsb = 14, .width = 3};
inline constexpr Field kIalu{.lsb = 11, .width = 3};
inline constexpr Field kC0Pred{.lsb = 10, .width = 1};
inline constexpr Field kXS1{.lsb = 9, .width = 1};
inline constexpr Field kDstS1{.lsb = 8, .width = 1};
inline constexpr Field kCtx{.lsb = 7, .width = 1};
inline constexpr Field kRm{.lsb = 5, .width = 2};

enum class Map : uint8_t { kNone = 0, kGroup = 1, kUpper = 2 };
enum class Pbuf : uint8_t { kNone = 0, kRowVec = 1, kTile = 2 };
enum class C0 : uint8_t { kZero = 0, kX2 = 1, kPbuf = 2, kRs0 = 3, kImm = 4 };
enum class B : uint8_t {
  kOne = 0,
  kX2 = 1,
  kPbuf = 2,
  kRs1 = 3,
  kImm = 4,
  kPre = 5,
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
  kRsqrt = 4,
};
enum class Cmp : uint8_t {
  kNone = 0,
  kLt = 1,
  kLe = 2,
  kEq = 3,
  kNe = 4,
  kGt = 5,
  kGe = 6,
};
enum class CmpLeft : uint8_t {
  kX = 0,
  kX2 = 1,
  kColumn = 2,
  kPosition = 3,
  kRowColumn = 4,
};
enum class Sel : uint8_t {
  kNone = 0,
  kZero = 1,
  kX2 = 2,
  kImm = 3,
  kNull = 4,
  kRs1 = 5,
};
enum class Ialu : uint8_t {
  kNone = 0,
  kAdd = 1,
  kSub = 2,
  kAnd = 3,
  kOr = 4,
  kXor = 5,
  kShr1 = 6,
  kRsub = 7,
};
enum class RowMask : uint8_t { kNone = 0, kSkip = 1, kSet = 2 };

inline constexpr uint32_t kPermPositionBits = 5;
inline constexpr uint32_t kPermSelectorBits = 4;
inline constexpr uint32_t kPermSame = 0;
inline constexpr uint32_t kPermZero = 6;
inline constexpr uint32_t kPermOne = 7;
inline constexpr uint32_t kPermInvert = 8;

constexpr uint32_t PermPosition(uint32_t position_bit) {
  return kPermPositionBits - position_bit;
}

}  // namespace pass

namespace transpose {

inline constexpr Field kRx{.lsb = 119, .width = 5};
inline constexpr Field kRsw{.lsb = 114, .width = 5};
inline constexpr Field kRd{.lsb = 109, .width = 5};
inline constexpr Field kRt{.lsb = 104, .width = 5};
inline constexpr Field kRdt{.lsb = 99, .width = 5};
inline constexpr Field kRst{.lsb = 94, .width = 5};
inline constexpr Field kRdw{.lsb = 89, .width = 5};
inline constexpr Field kRor{.lsb = 87, .width = 2};

}  // namespace transpose

namespace trace {

inline constexpr uint32_t kRecordBytes = 16;

inline constexpr Field kIndex{.lsb = 0, .width = 32};
inline constexpr Field kType{.lsb = 116, .width = 4};

inline constexpr uint32_t kTypeCompletion = 0;
inline constexpr uint32_t kTypeStatistics = 1;

inline constexpr Field kIssue{.lsb = 32, .width = 32};
inline constexpr Field kCompletion{.lsb = 64, .width = 32};
inline constexpr Field kSource{.lsb = 96, .width = 4};
inline constexpr Field kOpcode{.lsb = 100, .width = 8};
inline constexpr Field kWait{.lsb = 108, .width = 4};

inline constexpr Field kStatisticsSource{.lsb = 32, .width = 4};
inline constexpr Field kStatisticsA{.lsb = 40, .width = 24};
inline constexpr Field kStatisticsB{.lsb = 64, .width = 24};
inline constexpr Field kStatisticsC{.lsb = 88, .width = 24};
inline constexpr Field kStatisticsD{.lsb = 120, .width = 8};

inline constexpr uint32_t kSourceSequencer = kEngines;

}  // namespace trace

namespace library {

inline constexpr uint32_t kAdd = 0;
inline constexpr uint32_t kMul = 1;
inline constexpr uint32_t kMax = 2;
inline constexpr uint32_t kAddP = 3;
inline constexpr uint32_t kMulP = 4;
inline constexpr uint32_t kScale = 5;
inline constexpr uint32_t kCopy = 6;
inline constexpr uint32_t kRelu = 7;
inline constexpr uint32_t kAct = 8;
inline constexpr uint32_t kActMul = 9;
inline constexpr uint32_t kRope = 10;
inline constexpr uint32_t kSoftmax = 12;
inline constexpr uint32_t kLayerNorm = 22;
inline constexpr uint32_t kRmsNorm = 34;

struct Program {
  std::string_view name;
  uint32_t entry = 0;
};

inline constexpr std::array<Program, 14> kPrograms = {{
    {.name = "ADD", .entry = kAdd},
    {.name = "MUL", .entry = kMul},
    {.name = "MAX", .entry = kMax},
    {.name = "ADDP", .entry = kAddP},
    {.name = "MULP", .entry = kMulP},
    {.name = "SCALE", .entry = kScale},
    {.name = "COPY", .entry = kCopy},
    {.name = "RELU", .entry = kRelu},
    {.name = "ACT", .entry = kAct},
    {.name = "ACT_MUL", .entry = kActMul},
    {.name = "ROPE", .entry = kRope},
    {.name = "SOFTMAX", .entry = kSoftmax},
    {.name = "LAYERNORM", .entry = kLayerNorm},
    {.name = "RMSNORM", .entry = kRmsNorm},
}};

}  // namespace library

}  // namespace hotarubi::isa

#endif  // HOTARUBI_ISA_HPP_
