#include "program.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hotarubi/isa.hpp"
#include "hotarubi/json.hpp"
#include "isa.hpp"

namespace hotarubi::compiler {
namespace {

constexpr std::array<std::string_view, 5> kEngineNames = {"DMA", "GEMM", "VEC",
                                                          "CTRL", "WAIT"};
constexpr std::array<std::string_view, kMemCount> kMemNames = {
    "ABUF", "WBUF", "FBUF", "PBUF", "TBUF", "CBUF"};
constexpr uint32_t kRowLimit = 1U << isa::dma::kRows.width;
constexpr uint64_t kStrideLimit = uint64_t{16} << isa::dma::kDdrStride.width;
constexpr uint32_t kTokLimit = 1U << isa::dma::kTok.width;
constexpr uint32_t kBeatBytes = 16;

size_t MemIndex(Mem mem) { return static_cast<size_t>(mem); }

uint32_t Depth(Mem mem) { return kMemDepth.at(MemIndex(mem)); }

Range Span(Mem mem, uint32_t lo, uint32_t hi) {
  return {.mem = mem, .lo = lo, .hi = hi};
}

void CheckRange(const Range& range) {
  Require(range.lo < range.hi && range.hi <= Depth(range.mem),
          std::format("bad range {}..{} in memory {}", range.lo, range.hi,
                      MemIndex(range.mem)));
}

bool RowFormat(DmaKind kind) {
  return kind == DmaKind::kRow || kind == DmaKind::kRowIndexed;
}

uint32_t OnchipWords(const Transfer& transfer) {
  const uint32_t words = transfer.rows * transfer.row_words;
  if (!RowFormat(transfer.kind)) {
    return words;
  }
  const bool abuf = transfer.mem == Mem::kAbuf;
  Require(
      words % 4 == 0 && transfer.tok > 0 && transfer.tok <= (abuf ? 32U : 8U),
      std::format("row-format transfer of {} words with tok {}", words,
                  transfer.tok));
  return words / 4 * (abuf ? transfer.tok : 4 * transfer.tok);
}

void CheckTransfer(const Transfer& transfer) {
  Require(transfer.rows >= 1 && transfer.rows < kRowLimit &&
              transfer.row_words >= 1 && transfer.row_words < kRowLimit &&
              transfer.ddr_stride < kStrideLimit &&
              transfer.ddr_stride % kBeatBytes == 0 &&
              transfer.ddr_offset % kBeatBytes == 0 && transfer.tok < kTokLimit,
          std::format("bad DMA transfer: offset {:#x} rows {} row_words {} "
                      "stride {} tok {}",
                      transfer.ddr_offset, transfer.rows, transfer.row_words,
                      transfer.ddr_stride, transfer.tok));
}

bool MultiPass(VecOp op) {
  return op == VecOp::kLayerNorm || op == VecOp::kRmsNorm ||
         op == VecOp::kAct || op == VecOp::kActMul || op == VecOp::kSoftmax ||
         op == VecOp::kRope;
}

bool ScratchInSrc2(VecOp op) {
  return op == VecOp::kLayerNorm || op == VecOp::kRmsNorm;
}

}  // namespace

Program::Scope::Scope(Program& program, std::string_view tag)
    : program_(program) {
  program_.PushScope(tag);
}

Program::Scope::~Scope() { program_.PopScope(); }

Program::Program(uint64_t base) : base_(base) {
  tags_.emplace_back();
  tag_lookup_.emplace("", 0);
  ResetTracking();
}

void Program::ResetTracking() {
  known_.fill(0);
  wait_count_ = 0;
  for (size_t m = 0; m < kMemCount; ++m) {
    const size_t depth = kMemDepth.at(m);
    Tracker& tracker = trackers_.at(m);
    tracker.write_index.assign(depth, -1);
    tracker.write_engine.assign(depth, -1);
    for (auto& reads : tracker.read_index) {
      reads.assign(depth, -1);
    }
  }
}

uint32_t Program::DmaAddress(uint64_t offset) const {
  const uint64_t address = base_ + offset;
  Require(address < (uint64_t{1} << 32) && address % kBeatBytes == 0,
          std::format("DDR address {:#x} is not a 16-byte unit below 4 GB",
                      address));
  return static_cast<uint32_t>(address / kBeatBytes);
}

int32_t Program::LatestWrite(const Tracker& tracker, const Range& range,
                             int engine) {
  int32_t latest = -1;
  for (uint32_t a = range.lo; a < range.hi; ++a) {
    if (tracker.write_engine[a] == engine) {
      latest = std::max(latest, tracker.write_index[a]);
    }
  }
  return latest;
}

int32_t Program::LatestRead(const Tracker& tracker, const Range& range,
                            int engine) {
  const auto& reads = tracker.read_index.at(engine);
  return *std::max_element(reads.begin() + range.lo, reads.begin() + range.hi);
}

uint32_t Program::Dependencies(int engine, const std::vector<Range>& reads,
                               const std::vector<Range>& writes) const {
  uint32_t need = 0;
  for (const auto* list : {&reads, &writes}) {
    for (const Range& range : *list) {
      CheckRange(range);
      const Tracker& tracker = trackers_.at(MemIndex(range.mem));
      for (int e = 0; e < kEngines; ++e) {
        if (e != engine && LatestWrite(tracker, range, e) >= known_.at(e)) {
          need |= 1U << e;
        }
      }
    }
  }
  for (const Range& range : writes) {
    const Tracker& tracker = trackers_.at(MemIndex(range.mem));
    for (int e = 0; e < kEngines; ++e) {
      if (e != engine && LatestRead(tracker, range, e) >= known_.at(e)) {
        need |= 1U << e;
      }
    }
  }
  return need;
}

uint32_t Program::Admit(int32_t index, Engine engine,
                        const std::vector<Range>& reads,
                        const std::vector<Range>& writes) {
  const int e = static_cast<int>(engine);
  uint32_t need = 0;
  if (e < kEngines) {
    need = Dependencies(e, reads, writes);
  } else if (engine == Engine::kCtrl) {
    need = (1U << kEngines) - 1;
  }
  uint32_t wait = 0;
  for (int other = 0; other < kEngines; ++other) {
    if ((need & (1U << other)) != 0) {
      wait |= isa::WaitBit(static_cast<uint32_t>(other));
      known_.at(other) = index;
    }
  }
  if (e < kEngines) {
    known_.at(e) = index;
  }
  if (need != 0) {
    ++wait_count_;
  }
  if (e < kEngines) {
    for (const Range& range : reads) {
      auto& reads_by_e = trackers_.at(MemIndex(range.mem)).read_index.at(e);
      std::fill(reads_by_e.begin() + range.lo, reads_by_e.begin() + range.hi,
                index);
    }
    for (const Range& range : writes) {
      Tracker& tracker = trackers_.at(MemIndex(range.mem));
      std::fill(tracker.write_index.begin() + range.lo,
                tracker.write_index.begin() + range.hi, index);
      std::fill(tracker.write_engine.begin() + range.lo,
                tracker.write_engine.begin() + range.hi,
                static_cast<int8_t>(e));
      for (auto& reads_by_engine : tracker.read_index) {
        std::fill(reads_by_engine.begin() + range.lo,
                  reads_by_engine.begin() + range.hi, -1);
      }
    }
  }
  return wait;
}

size_t Program::Emit(Engine engine,
                     const std::function<Instruction(uint32_t)>& make,
                     const std::vector<Range>& reads,
                     const std::vector<Range>& writes, std::string_view note) {
  const auto index = static_cast<int32_t>(instructions_.size());
  const uint32_t wait = Admit(index, engine, reads, writes);
  instructions_.push_back(make(wait));
  notes_.emplace_back(note);
  engines_.push_back(engine);
  reads_.push_back(reads);
  writes_.push_back(writes);
  tag_index_.push_back(tag_lookup_.at(scope_));
  return static_cast<size_t>(index);
}

void Program::Reorder(std::span<const size_t> order) {
  const size_t count = instructions_.size();
  std::vector<size_t> sorted(order.begin(), order.end());
  std::ranges::sort(sorted);
  std::vector<size_t> identity(count);
  std::ranges::iota(identity, size_t{0});
  Require(order.size() == count && sorted == identity,
          "the order is not a permutation of the instructions");
  const auto permute = [&](auto& items) {
    auto reordered = items;
    for (size_t i = 0; i < count; ++i) {
      reordered[i] = std::move(items[order[i]]);
    }
    items = std::move(reordered);
  };
  permute(instructions_);
  permute(notes_);
  permute(engines_);
  permute(reads_);
  permute(writes_);
  permute(tag_index_);
  ResetTracking();
  for (size_t i = 0; i < count; ++i) {
    const uint32_t wait =
        Admit(static_cast<int32_t>(i), engines_[i], reads_[i], writes_[i]);
    instructions_[i].Set(isa::kWait, wait);
  }
}

size_t Program::Load(const Transfer& transfer, std::string_view note) {
  CheckTransfer(transfer);
  Require(transfer.kind != DmaKind::kGacc, "a GACC is emitted by Gacc");
  const uint32_t words = OnchipWords(transfer);
  const DmaFields fields{.ddr_addr = DmaAddress(transfer.ddr_offset),
                         .mem = transfer.mem,
                         .onchip = transfer.onchip,
                         .rows = transfer.rows,
                         .row_words = transfer.row_words,
                         .ddr_stride = transfer.ddr_stride / kBeatBytes,
                         .kind = transfer.kind,
                         .tok = transfer.tok};
  return Emit(
      Engine::kDma, [&](uint32_t wait) { return EncodeDmaLoad(fields, wait); },
      {}, {Span(transfer.mem, transfer.onchip, transfer.onchip + words)}, note);
}

size_t Program::Store(const Transfer& transfer, std::string_view note) {
  CheckTransfer(transfer);
  Require(transfer.kind == DmaKind::kRaw || transfer.kind == DmaKind::kRow,
          "a store is a RAW or ROW transfer");
  const uint32_t words = OnchipWords(transfer);
  const DmaFields fields{.ddr_addr = DmaAddress(transfer.ddr_offset),
                         .mem = transfer.mem,
                         .onchip = transfer.onchip,
                         .rows = transfer.rows,
                         .row_words = transfer.row_words,
                         .ddr_stride = transfer.ddr_stride / kBeatBytes,
                         .kind = transfer.kind,
                         .tok = transfer.tok};
  return Emit(
      Engine::kDma, [&](uint32_t wait) { return EncodeDmaStore(fields, wait); },
      {Span(transfer.mem, transfer.onchip, transfer.onchip + words)}, {}, note);
}

size_t Program::Gacc(const GaccArgs& args, std::string_view note) {
  Require(args.slots >= 1 && args.slots < kRowLimit && args.blocks >= 1 &&
              args.blocks <= 8 && args.list_offset % kBeatBytes == 0 &&
              args.coefficient_stride % kBeatBytes == 0 &&
              args.coefficient_stride < kStrideLimit,
          std::format("bad GACC: slots {} blocks {} coefficient stride {}",
                      args.slots, args.blocks, args.coefficient_stride));
  const DmaFields fields{.ddr_addr = DmaAddress(args.list_offset),
                         .mem = Mem::kFbuf,
                         .onchip = args.dst,
                         .rows = args.slots,
                         .row_words = args.blocks,
                         .ddr_stride = args.coefficient_stride / kBeatBytes,
                         .kind = DmaKind::kGacc,
                         .tok = 0,
                         .acc = args.acc,
                         .flush = args.flush};
  std::vector<Range> writes;
  if (args.flush) {
    writes.push_back(Span(Mem::kFbuf, args.dst, args.dst + (4 * args.blocks)));
  }
  return Emit(
      Engine::kDma, [&](uint32_t wait) { return EncodeDmaLoad(fields, wait); },
      {}, writes, note);
}

size_t Program::Gemm(const GemmArgs& args, std::string_view note) {
  Require(args.m % 32 == 0 && args.m >= 32 && args.m <= 992 && args.kb >= 1 &&
              args.kb <= args.a_stride && args.n8 >= 1 && args.n8 <= 255 &&
              args.w_stride >= args.kb,
          std::format("bad GEMM: m {} kb {} a_stride {} n8 {} w_stride {}",
                      args.m, args.kb, args.a_stride, args.n8, args.w_stride));
  const uint32_t quads = args.m / 4;
  std::vector<Range> reads = {
      Span(Mem::kAbuf, args.a_base,
           args.a_base + ((quads - 1) * args.a_stride) + args.kb),
      Span(Mem::kWbuf, args.w_base,
           args.w_base + ((args.n8 - 1) * args.w_stride) + args.kb)};
  const Range output =
      Span(Mem::kFbuf, args.c_base, args.c_base + (quads * args.n8));
  if (args.acc) {
    reads.push_back(output);
  }
  const GemmFields fields{.a_base = args.a_base,
                          .a_stride = args.a_stride,
                          .kb = args.kb,
                          .m = args.m,
                          .w_base = args.w_base,
                          .w_stride = args.w_stride,
                          .n8 = args.n8,
                          .c_base = args.c_base,
                          .wmode = args.wmode,
                          .acc = args.acc};
  return Emit(
      Engine::kGemm, [&](uint32_t wait) { return EncodeGemm(fields, wait); },
      reads, {output}, note);
}

size_t Program::Vec(const VecArgs& args, std::string_view note) {
  const VecOp op = args.op;
  const uint32_t n = args.rows_q * args.n8;
  uint32_t src2 = args.src2;
  uint32_t imm = args.imm;
  uint32_t aux = args.aux;
  std::vector<Range> reads = {Span(Mem::kFbuf, args.src, args.src + n)};
  std::vector<Range> writes;
  if (op == VecOp::kAdd || op == VecOp::kMul || op == VecOp::kActMul ||
      op == VecOp::kMax) {
    reads.push_back(Span(Mem::kFbuf, src2, src2 + n));
  }
  if (op == VecOp::kAddP || op == VecOp::kMulP || op == VecOp::kRmsNorm) {
    reads.push_back(
        Span(Mem::kPbuf, args.p_base, args.p_base + ((args.n8 + 3) / 4)));
  }
  if (op == VecOp::kLayerNorm) {
    reads.push_back(Span(Mem::kPbuf, args.p_base, args.p_base + (args.n8 / 2)));
  }
  if (op == VecOp::kRope) {
    reads.push_back(
        Span(Mem::kPbuf, args.p_base, args.p_base + (8 * args.rows_q)));
  }
  if (op == VecOp::kAct || op == VecOp::kActMul || op == VecOp::kSoftmax) {
    reads.push_back(Span(Mem::kTbuf, 0, Depth(Mem::kTbuf)));
  }
  if ((op == VecOp::kLayerNorm || op == VecOp::kRmsNorm) && aux == 0) {
    aux = 8 * args.n8;
  }
  uint32_t scratch = args.dst;
  if (args.q) {
    Require(args.n8 % 4 == 0 &&
                args.dst + (args.rows_q * (args.n8 / 4)) <= Depth(Mem::kAbuf),
            std::format("bad quantized VEC output at {}", args.dst));
    writes.push_back(
        Span(Mem::kAbuf, args.dst, args.dst + (args.rows_q * (args.n8 / 4))));
    if (MultiPass(op)) {
      Require(args.scratch.has_value(),
              "a multi-pass VEC operation with Q needs a scratch area");
      scratch = args.scratch.value_or(0);
      reads.push_back(Span(Mem::kFbuf, scratch, scratch + n));
      writes.push_back(Span(Mem::kFbuf, scratch, scratch + n));
    }
  } else {
    writes.push_back(Span(Mem::kFbuf, args.dst, args.dst + n));
    if (MultiPass(op)) {
      reads.push_back(Span(Mem::kFbuf, args.dst, args.dst + n));
    }
    Require(op != VecOp::kRope || args.dst != args.src,
            "ROPE needs a destination different from its source");
  }
  if (MultiPass(op)) {
    (ScratchInSrc2(op) ? src2 : imm) = scratch;
  }
  const VecFields fields{.entry = static_cast<uint32_t>(op),
                         .src = args.src,
                         .src2 = src2,
                         .dst = args.dst,
                         .rows_q = args.rows_q,
                         .n8 = args.n8,
                         .p_base = args.p_base,
                         .aux = aux,
                         .imm = imm,
                         .q = args.q};
  return Emit(
      Engine::kVec, [&](uint32_t wait) { return EncodeVec(fields, wait); },
      reads, writes, note);
}

size_t Program::Run(const RunArgs& args, std::string_view note) {
  Require(args.pc % isa::vec::kEntryWords == 0 &&
              args.pc >= isa::vec::kRomEntries * isa::vec::kEntryWords,
          std::format("a loaded VEC program starts at CBUF word {}", args.pc));
  const Range whole = Span(Mem::kFbuf, 0, Depth(Mem::kFbuf));
  std::vector<Range> reads = args.reads.value_or(
      std::vector<Range>{whole, Span(Mem::kPbuf, 0, Depth(Mem::kPbuf)),
                         Span(Mem::kTbuf, 0, Depth(Mem::kTbuf))});
  reads.push_back(Span(Mem::kCbuf, 0, Depth(Mem::kCbuf)));
  const std::vector<Range> writes =
      args.writes.value_or(std::vector<Range>{whole});
  const VecFields fields{.entry = args.pc / isa::vec::kEntryWords,
                         .src = args.src,
                         .src2 = args.src2,
                         .dst = args.dst,
                         .rows_q = args.rows_q,
                         .n8 = args.n8,
                         .p_base = args.p_base,
                         .aux = 0,
                         .imm = args.imm,
                         .q = false};
  return Emit(
      Engine::kVec, [&](uint32_t wait) { return EncodeVec(fields, wait); },
      reads, writes, note);
}

size_t Program::Signal(uint32_t id, uint32_t value, std::string_view note) {
  return Emit(
      Engine::kCtrl,
      [&](uint32_t wait) { return EncodeSignal(id, value, wait); }, {}, {},
      note);
}

size_t Program::WaitHost(uint32_t value, std::string_view note) {
  return Emit(
      Engine::kWait, [&](uint32_t wait) { return EncodeWaitHost(value, wait); },
      {}, {}, note);
}

size_t Program::Halt(std::string_view note) {
  return Emit(
      Engine::kCtrl, [&](uint32_t wait) { return EncodeHalt(wait); }, {}, {},
      note);
}

size_t Program::EngineCount(Engine engine) const {
  return static_cast<size_t>(
      std::count(engines_.begin(), engines_.end(), engine));
}

void Program::SetDdrOffset(size_t index, uint64_t ddr_offset) {
  Require(index < instructions_.size() && engines_[index] == Engine::kDma &&
              ddr_offset % kBeatBytes == 0,
          std::format("instruction {} is not a DMA transfer", index));
  instructions_[index].Set(isa::dma::kDdrAddr, DmaAddress(ddr_offset));
}

void Program::PushScope(std::string_view tag) {
  const size_t length = scope_.size();
  if (!scope_.empty()) {
    scope_ += '/';
  }
  scope_ += tag;
  scope_lengths_.push_back(length);
  if (!tag_lookup_.contains(scope_)) {
    tag_lookup_.emplace(scope_, static_cast<uint32_t>(tags_.size()));
    tags_.push_back(scope_);
  }
}

void Program::PopScope() {
  Require(!scope_lengths_.empty(), "a scope is closed that was not opened");
  scope_.resize(scope_lengths_.back());
  scope_lengths_.pop_back();
}

std::string_view Program::Tag(size_t index) const {
  return tags_.at(tag_index_.at(index));
}

std::vector<std::byte> Program::Bytes() const {
  return Serialize(instructions_);
}

std::string Program::Listing() const {
  std::string text;
  for (size_t i = 0; i < instructions_.size(); ++i) {
    const Instruction& instruction = instructions_[i];
    text +=
        std::format("{:5d} {:4s} w={:x} {} {}\n", i,
                    kEngineNames.at(static_cast<size_t>(engines_[i])),
                    instruction.Get(isa::kWait), instruction.Hex(), notes_[i]);
  }
  return text;
}

std::string Program::ListingJson() const {
  const auto ranges = [](const std::vector<Range>& list) {
    std::string text = "[";
    for (size_t i = 0; i < list.size(); ++i) {
      text += std::format("{}[\"{}\",{},{}]", i == 0 ? "" : ",",
                          kMemNames.at(MemIndex(list[i].mem)), list[i].lo,
                          list[i].hi);
    }
    return text + "]";
  };
  std::string text = std::format(R"({{"base":{},"tags":[)", base_);
  for (size_t i = 0; i < tags_.size(); ++i) {
    text += (i == 0 ? "" : ",") + Json(tags_[i]).Dump();
  }
  text += "],\"instructions\":[\n";
  for (size_t i = 0; i < instructions_.size(); ++i) {
    const Instruction& instruction = instructions_[i];
    text += std::format(
        "{}{{\"index\":{},\"engine\":\"{}\",\"opcode\":{},\"wait\":{},"
        "\"hex\":\"{}\",\"tag\":{},\"note\":{},\"reads\":{},"
        "\"writes\":{}}}",
        i == 0 ? "" : ",\n", i,
        kEngineNames.at(static_cast<size_t>(engines_[i])),
        instruction.Get(isa::kOpcode), instruction.Get(isa::kWait),
        instruction.Hex(), tag_index_[i], Json(notes_[i]).Dump(),
        ranges(reads_[i]), ranges(writes_[i]));
  }
  return text + "\n]}\n";
}

}  // namespace hotarubi::compiler
