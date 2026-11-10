#include "isa_pkg.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>

#include "hotarubi/driver.hpp"
#include "hotarubi/isa.hpp"

namespace hotarubi::compiler::isa_pkg {
namespace {

template <typename E>
constexpr uint32_t Code(E value) {
  return static_cast<uint32_t>(value);
}

constexpr uint32_t BitIndex(uint32_t mask) {
  return static_cast<uint32_t>(std::countr_zero(mask));
}

static_assert(isa::transpose::kRx.lsb == isa::pass::kRx.lsb &&
              isa::transpose::kRsw.lsb == isa::pass::kRx2.lsb &&
              isa::transpose::kRd.lsb == isa::pass::kRd.lsb &&
              isa::transpose::kRt.lsb == isa::pass::kRq.lsb &&
              isa::transpose::kRdt.lsb == isa::pass::kRnw.lsb &&
              isa::transpose::kRst.lsb == isa::pass::kRn8.lsb &&
              isa::transpose::kRdw.lsb == isa::pass::kRpb.lsb);
static_assert(isa::vec::kRegSrc2 == isa::vec::kRegSrc + 1 &&
              isa::vec::kRegDst == isa::vec::kRegSrc + 2 &&
              isa::vec::kRegRowsQ == isa::vec::kRegSrc + 3 &&
              isa::vec::kRegN8 == isa::vec::kRegSrc + 4 &&
              isa::vec::kRegPBase == isa::vec::kRegSrc + 5 &&
              isa::vec::kRegAux == isa::vec::kRegSrc + 6 &&
              isa::vec::kRegImm == isa::vec::kRegSrc + 7 &&
              isa::vec::kRegQ == isa::vec::kRegSrc + 8);

template <typename E>
using Member = std::pair<std::string_view, E>;

class Writer {
 public:
  void Blank() { text_ += "\n"; }

  void Int(std::string_view name, uint32_t value) {
    Line(std::format("localparam int {} = {};", name, value));
  }

  void Field(std::string_view name, isa::Field field) {
    Int(std::format("{}Msb", name), field.lsb + field.width - 1);
    Int(std::format("{}Lsb", name), field.lsb);
  }

  void Value(std::string_view name, uint32_t width, uint32_t value) {
    Line(std::format("localparam logic {}{} = {};", Range(width), name,
                     Literal(width, value)));
  }

  template <typename E>
  void Enum(std::string_view type, std::string_view prefix, uint32_t width,
            std::initializer_list<Member<E>> members) {
    Line(std::format("typedef enum logic {}{{", Range(width)));
    size_t left = members.size();
    for (const Member<E>& member : members) {
      --left;
      text_ += std::format("    {}{} = {}{}\n", prefix, member.first,
                           Literal(width, Code(member.second)),
                           left == 0 ? "" : ",");
    }
    Line(std::format("}} {};", type));
  }

  std::string Finish() && {
    return std::format("package isa_pkg;\n\n{}\nendpackage\n", text_);
  }

 private:
  void Line(std::string_view line) { text_ += std::format("  {}\n", line); }

  static std::string Range(uint32_t width) {
    return width == 1 ? std::string() : std::format("[{}:0] ", width - 1);
  }

  static std::string Literal(uint32_t width, uint32_t value) {
    return width >= 16 ? std::format("{}'h{:X}", width, value)
                       : std::format("{}'d{}", width, value);
  }

  std::string text_;
};

void Instructions(Writer& w) {
  namespace d = isa::dma;
  namespace g = isa::gemm;
  namespace v = isa::vec;
  w.Field("Opcode", isa::kOpcode);
  w.Field("Wait", isa::kWait);
  w.Blank();

  const uint32_t op = isa::kOpcode.width;
  w.Value("OpNop", op, isa::kOpNop);
  w.Value("OpDmaLoad", op, isa::kOpDmaLoad);
  w.Value("OpDmaStore", op, isa::kOpDmaStore);
  w.Value("OpGemm", op, isa::kOpGemm);
  w.Value("OpVec", op, isa::kOpVec);
  w.Value("OpSignal", op, isa::kOpSignal);
  w.Value("OpHalt", op, isa::kOpHalt);
  w.Value("OpWaitHost", op, isa::kOpWaitHost);
  w.Blank();

  w.Int("Engines", isa::kEngines);
  w.Int("EngDma", isa::kEngineDma);
  w.Int("EngGemm", isa::kEngineGemm);
  w.Int("EngVec", isa::kEngineVec);
  w.Blank();

  const uint32_t mem = d::kMem.width;
  w.Value("MemAbuf", mem, Code(isa::Mem::kAbuf));
  w.Value("MemWbuf", mem, Code(isa::Mem::kWbuf));
  w.Value("MemFbuf", mem, Code(isa::Mem::kFbuf));
  w.Value("MemPbuf", mem, Code(isa::Mem::kPbuf));
  w.Value("MemTbuf", mem, Code(isa::Mem::kTbuf));
  w.Value("MemCbuf", mem, Code(isa::Mem::kCbuf));
  w.Value("NullEntry", 32, isa::kNullEntry);
  w.Blank();

  w.Field("DmaDdrAddr", d::kDdrAddr);
  w.Field("DmaMem", d::kMem);
  w.Field("DmaAddr", d::kAddr);
  w.Field("DmaRows", d::kRows);
  w.Field("DmaRowWords", d::kRowWords);
  w.Field("DmaDdrStride", d::kDdrStride);
  w.Field("DmaKind", d::kKind);
  w.Field("DmaTok", d::kTok);
  w.Field("DmaAcc", d::kAcc);
  w.Field("DmaFlush", d::kFlush);
  const uint32_t kind = d::kKind.width;
  w.Value("DmaKindRaw", kind, d::kKindRaw);
  w.Value("DmaKindRow", kind, d::kKindRow);
  w.Value("DmaKindRowIndexed", kind, d::kKindRowIndexed);
  w.Value("DmaKindSplit8", kind, d::kKindSplit8);
  w.Value("DmaKindT32Split8", kind, d::kKindT32Split8);
  w.Value("DmaKindGacc", kind, d::kKindGacc);
  w.Blank();

  w.Field("GemmABase", g::kABase);
  w.Field("GemmAStride", g::kAStride);
  w.Field("GemmKb", g::kKb);
  w.Field("GemmM", g::kM);
  w.Field("GemmWBase", g::kWBase);
  w.Field("GemmWStride", g::kWStride);
  w.Field("GemmN8", g::kN8);
  w.Field("GemmCBase", g::kCBase);
  w.Field("GemmWmode", g::kWmode);
  w.Field("GemmAcc", g::kAcc);
  const uint32_t wmode = g::kWmode.width;
  w.Value("WeightModeFp4", wmode, Code(g::WeightMode::kFp4));
  w.Value("WeightModeW8Hi", wmode, Code(g::WeightMode::kW8Hi));
  w.Value("WeightModeW8Lo", wmode, Code(g::WeightMode::kW8Lo));
  w.Blank();

  w.Field("VecEntry", v::kEntry);
  w.Field("VecQ", v::kQ);
  w.Field("VecSrc", v::kSrc);
  w.Field("VecSrc2", v::kSrc2);
  w.Field("VecDst", v::kDst);
  w.Field("VecRowsQ", v::kRowsQ);
  w.Field("VecN8", v::kN8);
  w.Field("VecPBase", v::kPBase);
  w.Field("VecAux", v::kAux);
  w.Field("VecImm", v::kImm);
  const uint32_t reg = isa::program::kRd.width;
  w.Value("VecRegSrc", reg, v::kRegSrc);
  w.Value("VecRegSrc2", reg, v::kRegSrc2);
  w.Value("VecRegDst", reg, v::kRegDst);
  w.Value("VecRegRowsQ", reg, v::kRegRowsQ);
  w.Value("VecRegN8", reg, v::kRegN8);
  w.Value("VecRegPBase", reg, v::kRegPBase);
  w.Value("VecRegAux", reg, v::kRegAux);
  w.Value("VecRegImm", reg, v::kRegImm);
  w.Value("VecRegQ", reg, v::kRegQ);
  w.Int("VecEntryWords", v::kEntryWords);
  w.Int("VecRomEntries", v::kRomEntries);
  w.Int("VecRomWords", v::kRomWords);
  w.Blank();

  w.Field("SignalId", isa::signal::kId);
  w.Field("SignalValue", isa::signal::kValue);
  w.Field("WaitHostValue", isa::wait_host::kValue);
}

void ProgramWords(Writer& w) {
  namespace pg = isa::program;
  w.Field("ProgKind", pg::kKind);
  const uint32_t kind = pg::kKind.width;
  w.Value("KindPass", kind, pg::kKindPass);
  w.Value("KindSalu", kind, pg::kKindSalu);
  w.Value("KindSmem", kind, pg::kKindSmem);
  w.Value("KindBranch", kind, pg::kKindBranch);
  w.Value("KindRet", kind, pg::kKindRet);
  w.Value("KindNop", kind, pg::kKindNop);
  w.Value("KindTranspose", kind, pg::kKindTranspose);
  w.Blank();

  w.Field("ProgOp", pg::kOp);
  w.Field("ProgRd", pg::kRd);
  w.Field("ProgRa", pg::kRa);
  w.Field("ProgRb", pg::kRb);
  w.Field("ProgUseImm", pg::kUseImm);
  w.Field("ProgImm", pg::kImm);
  w.Field("ProgSmemCol", pg::kSmemCol);
  w.Field("ProgSmemWhich", pg::kSmemWhich);
  w.Field("ProgSmemCtx", pg::kSmemCtx);
  w.Value("ProgRegZero", pg::kRd.width, pg::kRegZero);
  w.Value("SmemWhichRs1", pg::kSmemWhich.width, pg::kSmemWhichRs1);
  w.Value("SmemWhichRs0", pg::kSmemWhich.width, pg::kSmemWhichRs0);
  w.Blank();

  const uint32_t op = pg::kOp.width;
  w.Value("SaluAdd", op, Code(pg::Salu::kAdd));
  w.Value("SaluSub", op, Code(pg::Salu::kSub));
  w.Value("SaluAnd", op, Code(pg::Salu::kAnd));
  w.Value("SaluOr", op, Code(pg::Salu::kOr));
  w.Value("SaluXor", op, Code(pg::Salu::kXor));
  w.Value("SaluShl", op, Code(pg::Salu::kShl));
  w.Value("SaluShr", op, Code(pg::Salu::kShr));
  w.Value("SaluSra", op, Code(pg::Salu::kSra));
  w.Value("SaluMul", op, Code(pg::Salu::kMul));
  w.Value("SaluSlt", op, Code(pg::Salu::kSlt));
  w.Value("SaluSltu", op, Code(pg::Salu::kSltu));
  w.Value("SaluSeq", op, Code(pg::Salu::kSeq));
  w.Value("SaluMov", op, Code(pg::Salu::kMov));
  w.Value("SmemSld", op, Code(pg::Smem::kSld));
  w.Value("SmemRsmov", op, Code(pg::Smem::kRsmov));
  w.Value("SmemRn", op, Code(pg::Smem::kRn));
  w.Value("BrAlways", op, Code(pg::Cond::kAlways));
  w.Value("BrZero", op, Code(pg::Cond::kZ));
  w.Value("BrNonzero", op, Code(pg::Cond::kNz));
}

void PassWords(Writer& w) {
  namespace ps = isa::pass;
  w.Field("PassRx", ps::kRx);
  w.Field("PassRx2", ps::kRx2);
  w.Field("PassRd", ps::kRd);
  w.Field("PassRq", ps::kRq);
  w.Field("PassRnw", ps::kRnw);
  w.Field("PassRn8", ps::kRn8);
  w.Field("PassRpb", ps::kRpb);
  w.Field("PassRimm", ps::kRimm);
  w.Field("PassRimm2", ps::kRimm2);
  w.Field("PassX2En", ps::kX2En);
  w.Field("PassMg", ps::kMg);
  w.Field("PassMs", ps::kMs);
  w.Field("PassMx", ps::kMx);
  w.Field("PassMx2", ps::kMx2);
  w.Field("PassMd", ps::kMd);
  w.Field("PassPa", ps::kPa);
  w.Field("PassC0", ps::kC0);
  w.Field("PassB", ps::kB);
  w.Field("PassNl", ps::kNl);
  w.Field("PassPo", ps::kPo);
  w.Field("PassPv", ps::kPv);
  w.Field("PassRed", ps::kRed);
  w.Field("PassWr", ps::kWr);
  w.Field("PassSf", ps::kSf);
  w.Field("PassSt", ps::kSt);
  w.Field("PassSrn", ps::kSrn);
  w.Field("PassSeps", ps::kSeps);
  w.Field("PassPx", ps::kPx);
  w.Field("PassCmp", ps::kCmp);
  w.Field("PassCl", ps::kCl);
  w.Field("PassCr", ps::kCr);
  w.Field("PassCmpInt", ps::kCmpInt);
  w.Field("PassSel", ps::kSel);
  w.Field("PassIalu", ps::kIalu);
  w.Field("PassC0Pred", ps::kC0Pred);
  w.Field("PassXS1", ps::kXS1);
  w.Field("PassDstS1", ps::kDstS1);
  w.Field("PassCtx", ps::kCtx);
  w.Field("PassRm", ps::kRm);
  w.Blank();

  w.Enum<ps::Map>("mm_e", "Mm", ps::kMx.width,
                  {{"None", ps::Map::kNone},
                   {"Group", ps::Map::kGroup},
                   {"Upper", ps::Map::kUpper}});
  w.Blank();
  w.Enum<ps::Pbuf>("pa_e", "Pa", ps::kPa.width,
                   {{"None", ps::Pbuf::kNone},
                    {"RowVec", ps::Pbuf::kRowVec},
                    {"Tile", ps::Pbuf::kTile}});
  w.Blank();
  w.Enum<ps::C0>("c0_e", "C0", ps::kC0.width,
                 {{"Zero", ps::C0::kZero},
                  {"X2", ps::C0::kX2},
                  {"Pbuf", ps::C0::kPbuf},
                  {"Rs0", ps::C0::kRs0},
                  {"Imm", ps::C0::kImm}});
  w.Blank();
  w.Enum<ps::B>("b_e", "B", ps::kB.width,
                {{"One", ps::B::kOne},
                 {"X2", ps::B::kX2},
                 {"Pbuf", ps::B::kPbuf},
                 {"Rs1", ps::B::kRs1},
                 {"Imm", ps::B::kImm},
                 {"Pre", ps::B::kPre}});
  w.Blank();
  w.Enum<ps::Nl>("nl_e", "Nl", ps::kNl.width,
                 {{"None", ps::Nl::kNone},
                  {"Act", ps::Nl::kAct},
                  {"Exp2", ps::Nl::kExp2},
                  {"Tab01", ps::Nl::kTab01}});
  w.Blank();
  w.Enum<ps::Post>("po_e", "Po", ps::kPo.width,
                   {{"Add", ps::Post::kAdd},
                    {"Sub", ps::Post::kSub},
                    {"Max", ps::Post::kMax},
                    {"Signfix", ps::Post::kSignfix}});
  w.Blank();
  w.Enum<ps::Reduce>("red_e", "Red", ps::kRed.width,
                     {{"None", ps::Reduce::kNone},
                      {"Sum", ps::Reduce::kSum},
                      {"Max", ps::Reduce::kMax}});
  w.Blank();
  w.Enum<ps::Write>("wr_e", "Wr", ps::kWr.width,
                    {{"None", ps::Write::kNone},
                     {"Fbuf", ps::Write::kFbuf},
                     {"Abuf", ps::Write::kAbuf},
                     {"Final", ps::Write::kFinal}});
  w.Blank();
  w.Enum<ps::ScalarFn>("sf_e", "Sf", ps::kSf.width,
                       {{"None", ps::ScalarFn::kNone},
                        {"Identity", ps::ScalarFn::kIdentity},
                        {"Negate", ps::ScalarFn::kNegate},
                        {"Reciprocal", ps::ScalarFn::kReciprocal},
                        {"Rsqrt", ps::ScalarFn::kRsqrt}});
  w.Blank();
  w.Enum<ps::Cmp>("cmp_e", "Cmp", ps::kCmp.width,
                  {{"None", ps::Cmp::kNone},
                   {"Lt", ps::Cmp::kLt},
                   {"Le", ps::Cmp::kLe},
                   {"Eq", ps::Cmp::kEq},
                   {"Ne", ps::Cmp::kNe},
                   {"Gt", ps::Cmp::kGt},
                   {"Ge", ps::Cmp::kGe}});
  w.Blank();
  w.Enum<ps::CmpLeft>("cl_e", "Cl", ps::kCl.width,
                      {{"X", ps::CmpLeft::kX},
                       {"X2", ps::CmpLeft::kX2},
                       {"Column", ps::CmpLeft::kColumn},
                       {"Position", ps::CmpLeft::kPosition},
                       {"RowColumn", ps::CmpLeft::kRowColumn}});
  w.Blank();
  w.Enum<ps::Sel>("sel_e", "Sel", ps::kSel.width,
                  {{"None", ps::Sel::kNone},
                   {"Zero", ps::Sel::kZero},
                   {"X2", ps::Sel::kX2},
                   {"Imm", ps::Sel::kImm},
                   {"Null", ps::Sel::kNull},
                   {"Rs1", ps::Sel::kRs1}});
  w.Blank();
  w.Enum<ps::Ialu>("ia_e", "Ialu", ps::kIalu.width,
                   {{"None", ps::Ialu::kNone},
                    {"Add", ps::Ialu::kAdd},
                    {"Sub", ps::Ialu::kSub},
                    {"And", ps::Ialu::kAnd},
                    {"Or", ps::Ialu::kOr},
                    {"Xor", ps::Ialu::kXor},
                    {"Shr1", ps::Ialu::kShr1},
                    {"Rsub", ps::Ialu::kRsub}});
  w.Blank();
  w.Enum<ps::RowMask>("rm_e", "Rm", ps::kRm.width,
                      {{"None", ps::RowMask::kNone},
                       {"Skip", ps::RowMask::kSkip},
                       {"Set", ps::RowMask::kSet}});
  w.Blank();

  const uint32_t selector = ps::kPermSelectorBits - 1;
  w.Int("PermSelectorBits", ps::kPermSelectorBits);
  w.Int("PermPositionBits", ps::kPermPositionBits);
  w.Int("PermInvertBit", BitIndex(ps::kPermInvert));
  w.Value("PermSame", selector, ps::kPermSame);
  for (uint32_t bit = ps::kPermPositionBits; bit-- > 0;) {
    w.Value(std::format("PermPos{}", bit), selector, ps::PermPosition(bit));
  }
  w.Value("PermZero", selector, ps::kPermZero);
  w.Value("PermOne", selector, ps::kPermOne);
  w.Blank();

  namespace tr = isa::transpose;
  w.Field("TransposeRx", tr::kRx);
  w.Field("TransposeRsw", tr::kRsw);
  w.Field("TransposeRd", tr::kRd);
  w.Field("TransposeRt", tr::kRt);
  w.Field("TransposeRdt", tr::kRdt);
  w.Field("TransposeRst", tr::kRst);
  w.Field("TransposeRdw", tr::kRdw);
  w.Field("TransposeRor", tr::kRor);
}

void Registers(Writer& w) {
  w.Int("RegAddrBits", kRegisterAddressBits);
  w.Int("RegByteBits", BitIndex(kRegisterBytes));
  const uint32_t index = kRegisterAddressBits - BitIndex(kRegisterBytes);
  const auto value = [&](std::string_view name, Register reg) {
    w.Value(name, index, Offset(reg) / kRegisterBytes);
  };
  value("RegCtrl", Register::kCtrl);
  value("RegStatus", Register::kStatus);
  value("RegInstrBase", Register::kInstrBase);
  value("RegInstrCount", Register::kInstrCount);
  value("RegIrqStatus", Register::kIrqStatus);
  value("RegIrqEnable", Register::kIrqEnable);
  value("RegSignal", Register::kSignal);
  value("RegSignalValue", Register::kSignalValue);
  value("RegIssued", Register::kIssued);
  value("RegCycles", Register::kCycles);
  value("RegDmaCycles", Register::kDmaCycles);
  value("RegGemmCycles", Register::kGemmCycles);
  value("RegVecCycles", Register::kVecCycles);
  value("RegStallCycles", Register::kStallCycles);
  value("RegStarveCycles", Register::kStarveCycles);
  value("RegTraceBase", Register::kTraceBase);
  value("RegTraceCtrl", Register::kTraceCtrl);
  value("RegTraceCount", Register::kTraceCount);
  value("RegHostGo", Register::kHostGo);
  w.Blank();

  w.Int("CtrlStart", BitIndex(kCtrlStart));
  w.Int("CtrlSoftReset", BitIndex(kCtrlSoftReset));
  w.Int("StatusRunning", BitIndex(kStatusRunning));
  w.Int("StatusHalted", BitIndex(kStatusHalted));
  w.Int("StatusError", BitIndex(kStatusError));
  w.Int("StatusWaitingHost", BitIndex(kStatusWaitingHost));
  w.Int("StatusBusyLsb", kStatusBusyShift);
  w.Int("StatusIbufEmpty", BitIndex(kStatusIbufEmpty));
  w.Int("IrqSignal", BitIndex(kIrqSignal));
  w.Int("IrqHalt", BitIndex(kIrqHalt));
  w.Int("IrqIllegal", BitIndex(kIrqIllegal));
  w.Int("IrqAxiError", BitIndex(kIrqAxiError));
  w.Int("IrqBits", static_cast<uint32_t>(std::bit_width(kIrqAll)));
  w.Field("SignalRegId", kSignalId);
  w.Field("SignalRegCount", kSignalCount);
  w.Int("TraceCtrlEnable", BitIndex(kTraceEnable));
  w.Int("TraceCtrlIdle", BitIndex(kTraceIdle));
  w.Int("TraceCtrlStatistics", BitIndex(kTraceStatistics));
}

void TraceRecords(Writer& w) {
  namespace t = isa::trace;
  w.Int("TraceRecordBits", t::kRecordBytes * 8);
  w.Field("TraceIndex", t::kIndex);
  w.Field("TraceType", t::kType);
  w.Field("TraceIssue", t::kIssue);
  w.Field("TraceCompletion", t::kCompletion);
  w.Field("TraceSource", t::kSource);
  w.Field("TraceOpcode", t::kOpcode);
  w.Field("TraceWait", t::kWait);
  w.Field("TraceStatSource", t::kStatisticsSource);
  w.Field("TraceStatA", t::kStatisticsA);
  w.Field("TraceStatB", t::kStatisticsB);
  w.Field("TraceStatC", t::kStatisticsC);
  w.Field("TraceStatD", t::kStatisticsD);
  w.Value("TraceTypeCompletion", t::kType.width, t::kTypeCompletion);
  w.Value("TraceTypeStatistics", t::kType.width, t::kTypeStatistics);
  w.Int("TraceSourceSequencer", t::kSourceSequencer);
}

}  // namespace

std::string Package() {
  Writer w;
  Instructions(w);
  w.Blank();
  ProgramWords(w);
  w.Blank();
  PassWords(w);
  w.Blank();
  Registers(w);
  w.Blank();
  TraceRecords(w);
  return std::move(w).Finish();
}

}  // namespace hotarubi::compiler::isa_pkg
