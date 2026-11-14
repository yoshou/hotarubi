#include "program.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "hotarubi/isa.hpp"
#include "hotarubi/json.hpp"
#include "hotarubi/require.hpp"
#include "isa.hpp"
#include "machine.hpp"
#include "schedule.hpp"
#include "timing.hpp"
#include "word.hpp"

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
constexpr int kEngines = static_cast<int>(isa::kEngines);

size_t MemIndex(Mem mem) { return static_cast<size_t>(mem); }

uint32_t Depth(Mem mem) { return kMemDepth.at(MemIndex(mem)); }

std::string_view MemName(Mem mem) { return kMemNames.at(MemIndex(mem)); }

bool Tiled(Mem mem) {
  return mem == Mem::kAbuf || mem == Mem::kWbuf || mem == Mem::kFbuf ||
         mem == Mem::kPbuf;
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

bool ReadsSrc2(VecOp op) {
  return op == VecOp::kAdd || op == VecOp::kMul || op == VecOp::kActMul ||
         op == VecOp::kMax;
}

bool ReadsPbuf(VecOp op) {
  return op == VecOp::kAddP || op == VecOp::kMulP || op == VecOp::kRmsNorm ||
         op == VecOp::kLayerNorm || op == VecOp::kRope;
}

void AddTableReads(const VecArgs& args, std::vector<Access>& reads) {
  const VecOp op = args.op;
  if (op == VecOp::kAddP || op == VecOp::kMulP || op == VecOp::kRmsNorm) {
    reads.push_back(
        {.mem = Mem::kPbuf, .at = args.p_base, .words = (args.n8 + 3) / 4});
  }
  if (op == VecOp::kLayerNorm) {
    reads.push_back(
        {.mem = Mem::kPbuf, .at = args.p_base, .words = args.n8 / 2});
  }
  if (op == VecOp::kRope) {
    reads.push_back(
        {.mem = Mem::kPbuf, .at = args.p_base, .words = 8 * args.rows_q});
  }
  if (op == VecOp::kAct || op == VecOp::kActMul || op == VecOp::kSoftmax) {
    reads.push_back(Fixed(Mem::kTbuf, 0, Depth(Mem::kTbuf)));
  }
}

bool SamePlace(Place a, Place b) {
  return a.tile == b.tile && a.offset == b.offset;
}

iss::Word128 Word(const Instruction& instruction) {
  const auto& w = instruction.words;
  return {.low = w[0] | (uint64_t{w[1]} << 32),
          .high = w[2] | (uint64_t{w[3]} << 32)};
}

class Estimator {
 public:
  explicit Estimator(const iss::TimingParameters& timing) : model_(timing) {}

  uint64_t Cycles(const iss::Word128& word) {
    if (word.Field(isa::kOpcode) != isa::kOpVec) {
      return iss::EstimateCycles(word, model_);
    }
    const iss::VecCommand command = iss::DecodeVec(word);
    const auto key = std::make_tuple(command.entry, command.quantize,
                                     command.rows_q, command.n8, command.aux);
    auto found = vec_.find(key);
    if (found == vec_.end()) {
      found = vec_.emplace(key, iss::EstimateCycles(word, model_)).first;
    }
    return found->second;
  }

 private:
  iss::CostModel model_;
  std::map<std::tuple<uint32_t, bool, uint32_t, uint32_t, uint32_t>, uint64_t>
      vec_;
};

class WaitBits {
 public:
  WaitBits() {
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

  uint32_t Admit(int32_t index, Engine engine, const std::vector<Range>& reads,
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
    if (need != 0) {
      ++count_;
    }
    if (e >= kEngines) {
      return wait;
    }
    known_.at(e) = index;
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
    return wait;
  }

  [[nodiscard]] size_t Count() const { return count_; }

 private:
  struct Tracker {
    std::vector<int32_t> write_index;
    std::vector<int8_t> write_engine;
    std::array<std::vector<int32_t>, kEngines> read_index;
  };

  [[nodiscard]] uint32_t Dependencies(int engine,
                                      const std::vector<Range>& reads,
                                      const std::vector<Range>& writes) const {
    uint32_t need = 0;
    for (const auto* list : {&reads, &writes}) {
      for (const Range& range : *list) {
        const Tracker& tracker = trackers_.at(MemIndex(range.mem));
        for (uint32_t a = range.lo; a < range.hi; ++a) {
          const int8_t writer = tracker.write_engine[a];
          if (writer >= 0 && writer != engine &&
              tracker.write_index[a] >= known_.at(writer)) {
            need |= 1U << writer;
          }
        }
      }
    }
    for (const Range& range : writes) {
      const Tracker& tracker = trackers_.at(MemIndex(range.mem));
      for (int e = 0; e < kEngines; ++e) {
        if (e == engine) {
          continue;
        }
        const auto& reads_by_e = tracker.read_index.at(e);
        const int32_t latest = *std::max_element(reads_by_e.begin() + range.lo,
                                                 reads_by_e.begin() + range.hi);
        if (latest >= known_.at(e)) {
          need |= 1U << e;
        }
      }
    }
    return need;
  }

  std::array<int32_t, kEngines> known_{};
  std::array<Tracker, kMemCount> trackers_;
  size_t count_ = 0;
};

}  // namespace

Access Whole(const Tile& tile) {
  return {.mem = tile.Memory(), .at = tile.At(), .words = tile.Words()};
}

Access Part(const Tile& tile, uint32_t offset, uint32_t words) {
  return {.mem = tile.Memory(), .at = tile.At(offset), .words = words};
}

Access Fixed(Mem mem, uint32_t address, uint32_t words) {
  return {.mem = mem, .at = Place::Fixed(address), .words = words};
}

Program::Scope::Scope(Program& program, std::string_view tag)
    : program_(program) {
  program_.PushScope(tag);
}

Program::Scope::~Scope() { program_.PopScope(); }

Program::Program(uint64_t base) : base_(base) {
  tag_names_.emplace_back();
  tag_lookup_.emplace("", 0);
}

Tile Program::Alloc(Mem mem, uint32_t words) {
  Require(!done_ && Tiled(mem) && words >= 1 && words <= Depth(mem),
          std::format("bad tile of {} words in {}", words, MemName(mem)));
  tiles_.push_back({.mem = mem, .words = words});
  return {static_cast<uint32_t>(tiles_.size() - 1), mem, words};
}

void Program::Check(const Access& access) const {
  const Place at = access.at;
  if (Tiled(access.mem)) {
    Require(
        at.tile < tiles_.size() && tiles_[at.tile].mem == access.mem,
        std::format("{} is accessed outside its tiles", MemName(access.mem)));
    Require(
        access.words >= 1 && at.offset + access.words <= tiles_[at.tile].words,
        std::format("bad range {}..{} of a {}-word tile in {}", at.offset,
                    at.offset + access.words, tiles_[at.tile].words,
                    MemName(access.mem)));
  } else {
    Require(at.tile == kNoTile,
            std::format("{} is addressed directly", MemName(access.mem)));
    Require(access.words >= 1 && at.offset + access.words <= Depth(access.mem),
            std::format("bad range {}..{} in {}", at.offset,
                        at.offset + access.words, MemName(access.mem)));
  }
}

uint32_t Program::DmaAddress(uint64_t offset) const {
  const uint64_t address = base_ + offset;
  Require(address < (uint64_t{1} << 32) && address % kBeatBytes == 0,
          std::format("DDR address {:#x} is not a 16-byte unit below 4 GB",
                      address));
  return static_cast<uint32_t>(address / kBeatBytes);
}

size_t Program::Emit(Node node) {
  Require(!done_, "an instruction is emitted after the program is finalized");
  for (const auto* list : {&node.reads, &node.writes}) {
    for (const Access& access : *list) {
      Check(access);
    }
  }
  node.tag = tag_lookup_.at(scope_);
  nodes_.push_back(std::move(node));
  return nodes_.size() - 1;
}

size_t Program::Dma(const Transfer& transfer, DmaFields fields, bool store,
                    std::vector<Access> reads, std::vector<Access> writes,
                    std::string_view note) {
  static_cast<void>(DmaAddress(transfer.ddr_offset));
  return Emit({.engine = Engine::kDma,
               .reads = std::move(reads),
               .writes = std::move(writes),
               .note = std::string(note),
               .dma = fields,
               .dma_onchip = transfer.onchip,
               .dma_store = store,
               .ddr_offset = transfer.ddr_offset});
}

size_t Program::Load(const Transfer& transfer, std::string_view note) {
  CheckTransfer(transfer);
  Require(transfer.kind != DmaKind::kGacc, "a GACC is emitted by Gacc");
  const uint32_t words = OnchipWords(transfer);
  return Dma(transfer,
             {.mem = transfer.mem,
              .rows = transfer.rows,
              .row_words = transfer.row_words,
              .ddr_stride = transfer.ddr_stride / kBeatBytes,
              .kind = transfer.kind,
              .tok = transfer.tok},
             false, {},
             {{.mem = transfer.mem, .at = transfer.onchip, .words = words}},
             note);
}

size_t Program::Store(const Transfer& transfer, std::string_view note) {
  CheckTransfer(transfer);
  Require(transfer.kind == DmaKind::kRaw || transfer.kind == DmaKind::kRow,
          "a store is a RAW or ROW transfer");
  const uint32_t words = OnchipWords(transfer);
  return Dma(transfer,
             {.mem = transfer.mem,
              .rows = transfer.rows,
              .row_words = transfer.row_words,
              .ddr_stride = transfer.ddr_stride / kBeatBytes,
              .kind = transfer.kind,
              .tok = transfer.tok},
             true,
             {{.mem = transfer.mem, .at = transfer.onchip, .words = words}}, {},
             note);
}

size_t Program::Gacc(const GaccArgs& args, std::string_view note) {
  Require(args.slots >= 1 && args.slots < kRowLimit && args.blocks >= 1 &&
              args.blocks <= 8 && args.list_offset % kBeatBytes == 0 &&
              args.coefficient_stride % kBeatBytes == 0 &&
              args.coefficient_stride < kStrideLimit,
          std::format("bad GACC: slots {} blocks {} coefficient stride {}",
                      args.slots, args.blocks, args.coefficient_stride));
  const Access output{
      .mem = Mem::kFbuf, .at = args.dst, .words = 4 * args.blocks};
  Check(output);
  std::vector<Access> writes;
  if (args.flush) {
    writes.push_back(output);
  }
  return Dma(
      {.ddr_offset = args.list_offset, .mem = Mem::kFbuf, .onchip = args.dst},
      {.mem = Mem::kFbuf,
       .rows = args.slots,
       .row_words = args.blocks,
       .ddr_stride = args.coefficient_stride / kBeatBytes,
       .kind = DmaKind::kGacc,
       .tok = 0,
       .acc = args.acc,
       .flush = args.flush},
      false, {}, std::move(writes), note);
}

size_t Program::Gemm(const GemmArgs& args, std::string_view note) {
  Require(args.m % 32 == 0 && args.m >= 32 && args.m <= 992 && args.kb >= 1 &&
              args.kb <= args.a_stride && args.n8 >= 1 && args.n8 <= 255 &&
              args.w_stride >= args.kb,
          std::format("bad GEMM: m {} kb {} a_stride {} n8 {} w_stride {}",
                      args.m, args.kb, args.a_stride, args.n8, args.w_stride));
  const uint32_t quads = args.m / 4;
  std::vector<Access> reads = {
      {.mem = Mem::kAbuf,
       .at = args.a_base,
       .words = ((quads - 1) * args.a_stride) + args.kb},
      {.mem = Mem::kWbuf,
       .at = args.w_base,
       .words = ((args.n8 - 1) * args.w_stride) + args.kb}};
  const Access output{
      .mem = Mem::kFbuf, .at = args.c_base, .words = quads * args.n8};
  if (args.acc) {
    reads.push_back(output);
  }
  const GemmFields fields{.a_stride = args.a_stride,
                          .kb = args.kb,
                          .m = args.m,
                          .w_stride = args.w_stride,
                          .n8 = args.n8,
                          .wmode = args.wmode,
                          .acc = args.acc};
  return Emit({.engine = Engine::kGemm,
               .make =
                   [fields, a = args.a_base, w = args.w_base, c = args.c_base](
                       const Resolver& resolve, uint32_t wait) {
                     GemmFields placed = fields;
                     placed.a_base = resolve(a);
                     placed.w_base = resolve(w);
                     placed.c_base = resolve(c);
                     return EncodeGemm(placed, wait);
                   },
               .reads = std::move(reads),
               .writes = {output},
               .note = std::string(note)});
}

size_t Program::Vec(const VecArgs& args, std::string_view note) {
  const VecOp op = args.op;
  const uint32_t n = args.rows_q * args.n8;
  std::vector<Access> reads = {{.mem = Mem::kFbuf, .at = args.src, .words = n}};
  std::vector<Access> writes;
  if (ReadsSrc2(op)) {
    reads.push_back({.mem = Mem::kFbuf, .at = args.src2, .words = n});
  }
  AddTableReads(args, reads);
  uint32_t aux = args.aux;
  if ((op == VecOp::kLayerNorm || op == VecOp::kRmsNorm) && aux == 0) {
    aux = 8 * args.n8;
  }
  Place scratch = args.dst;
  if (args.q) {
    Require(args.n8 % 4 == 0,
            std::format("a quantized VEC output of {} words", args.n8));
    writes.push_back({.mem = Mem::kAbuf,
                      .at = args.dst,
                      .words = args.rows_q * (args.n8 / 4)});
    if (MultiPass(op)) {
      Require(args.scratch.has_value(),
              "a multi-pass VEC operation with Q needs a scratch area");
      scratch = args.scratch.value_or(Place{});
      reads.push_back({.mem = Mem::kFbuf, .at = scratch, .words = n});
      writes.push_back({.mem = Mem::kFbuf, .at = scratch, .words = n});
    }
  } else {
    writes.push_back({.mem = Mem::kFbuf, .at = args.dst, .words = n});
    if (MultiPass(op)) {
      reads.push_back({.mem = Mem::kFbuf, .at = args.dst, .words = n});
    }
    Require(op != VecOp::kRope || !SamePlace(args.dst, args.src),
            "ROPE needs a destination different from its source");
  }
  const VecFields fields{.entry = static_cast<uint32_t>(op),
                         .rows_q = args.rows_q,
                         .n8 = args.n8,
                         .aux = aux,
                         .imm = args.imm,
                         .q = args.q};
  const bool src2 = ReadsSrc2(op) || (MultiPass(op) && ScratchInSrc2(op));
  const bool imm = MultiPass(op) && !ScratchInSrc2(op);
  return Emit({.engine = Engine::kVec,
               .make =
                   [fields, args, scratch, src2, imm](const Resolver& resolve,
                                                      uint32_t wait) {
                     VecFields placed = fields;
                     placed.src = resolve(args.src);
                     placed.dst = resolve(args.dst);
                     if (src2) {
                       placed.src2 =
                           resolve(ScratchInSrc2(args.op) && MultiPass(args.op)
                                       ? scratch
                                       : args.src2);
                     }
                     if (imm) {
                       placed.imm = resolve(scratch);
                     }
                     if (ReadsPbuf(args.op)) {
                       placed.p_base = resolve(args.p_base);
                     }
                     return EncodeVec(placed, wait);
                   },
               .reads = std::move(reads),
               .writes = std::move(writes),
               .note = std::string(note)});
}

size_t Program::Run(const RunArgs& args, std::string_view note) {
  Require(args.pc % isa::vec::kEntryWords == 0 &&
              args.pc >= isa::vec::kRomEntries * isa::vec::kEntryWords,
          std::format("a loaded VEC program starts at CBUF word {}", args.pc));
  std::vector<Place> named = {args.src, args.src2, args.dst, args.p_base};
  for (const Packed& packed : args.imm_places) {
    named.push_back(packed.place);
  }
  for (const Place& place : named) {
    Require(place.tile == kNoTile ||
                std::ranges::any_of(
                    args.reads,
                    [&](const Access& a) { return a.at.tile == place.tile; }) ||
                std::ranges::any_of(
                    args.writes,
                    [&](const Access& a) { return a.at.tile == place.tile; }),
            "a loaded VEC program names a tile it does not declare");
  }
  std::vector<Access> reads = args.reads;
  reads.push_back(Fixed(Mem::kCbuf, 0, Depth(Mem::kCbuf)));
  const VecFields fields{.entry = args.pc / isa::vec::kEntryWords,
                         .rows_q = args.rows_q,
                         .n8 = args.n8,
                         .aux = 0,
                         .imm = args.imm,
                         .q = false};
  return Emit({.engine = Engine::kVec,
               .make =
                   [fields, args](const Resolver& resolve, uint32_t wait) {
                     VecFields placed = fields;
                     placed.src = resolve(args.src);
                     placed.src2 = resolve(args.src2);
                     placed.dst = resolve(args.dst);
                     placed.p_base = resolve(args.p_base);
                     for (const Packed& packed : args.imm_places) {
                       placed.imm |= resolve(packed.place) << packed.shift;
                     }
                     return EncodeVec(placed, wait);
                   },
               .reads = std::move(reads),
               .writes = args.writes,
               .note = std::string(note),
               .loaded = true});
}

size_t Program::Signal(uint32_t id, uint32_t value, std::string_view note) {
  return Emit({.engine = Engine::kCtrl,
               .make =
                   [id, value](const Resolver&, uint32_t wait) {
                     return EncodeSignal(id, value, wait);
                   },
               .reads = {},
               .writes = {},
               .note = std::string(note)});
}

size_t Program::WaitHost(uint32_t value, std::string_view note) {
  return Emit(
      {.engine = Engine::kWait,
       .make = [value](const Resolver&,
                       uint32_t wait) { return EncodeWaitHost(value, wait); },
       .reads = {},
       .writes = {},
       .note = std::string(note)});
}

size_t Program::Halt(std::string_view note) {
  return Emit(
      {.engine = Engine::kCtrl,
       .make = [](const Resolver&, uint32_t wait) { return EncodeHalt(wait); },
       .reads = {},
       .writes = {},
       .note = std::string(note)});
}

void Program::SetDdrOffset(size_t index, uint64_t ddr_offset) {
  Require(!done_ && index < nodes_.size() && nodes_[index].dma.has_value() &&
              ddr_offset % kBeatBytes == 0,
          std::format("instruction {} is not a DMA transfer", index));
  static_cast<void>(DmaAddress(ddr_offset));
  nodes_[index].ddr_offset = ddr_offset;
}

Instruction Program::Encode(const Node& node, const Resolver& resolve,
                            uint32_t wait) const {
  if (!node.dma) {
    return node.make(resolve, wait);
  }
  DmaFields fields = *node.dma;
  fields.ddr_addr = DmaAddress(node.ddr_offset);
  fields.onchip = resolve(node.dma_onchip);
  return node.dma_store ? EncodeDmaStore(fields, wait)
                        : EncodeDmaLoad(fields, wait);
}

std::vector<Step> Program::Steps(const iss::TimingParameters& timing) const {
  Estimator estimator(timing);
  const Resolver unplaced = [](Place place) { return place.offset; };
  std::vector<Step> steps;
  steps.reserve(nodes_.size());
  for (const Node& node : nodes_) {
    Step step{.engine = static_cast<int>(node.engine),
              .loaded = node.loaded,
              .spans = {}};
    if (step.engine < kEngines) {
      const iss::Word128 word = Word(Encode(node, unplaced, 0));
      step.cost = std::max<uint64_t>(estimator.Cycles(word), 1);
      if (node.dma) {
        step.dma = true;
        step.store = node.dma_store;
        step.extent = iss::EstimateDdrExtent(iss::DecodeDma(word));
      }
    }
    for (const bool write : {false, true}) {
      for (const Access& access : write ? node.writes : node.reads) {
        step.spans.push_back({.tile = access.at.tile,
                              .mem = access.mem,
                              .lo = access.at.offset,
                              .hi = access.at.offset + access.words,
                              .write = write});
      }
    }
    steps.push_back(std::move(step));
  }
  return steps;
}

void Program::Finalize(const ScheduleOptions& options) {
  Require(!done_, "the program is finalized twice");
  const Plan plan = ScheduleStream(Steps(options.timing), tiles_, options);
  Done done;
  done.base = plan.base;
  const Resolver resolve = [&done](Place place) {
    if (place.tile == kNoTile) {
      return place.offset;
    }
    Require(place.tile < done.base.size() && done.base[place.tile] != kNoTile,
            "an instruction names a tile that no instruction accesses");
    return done.base[place.tile] + place.offset;
  };
  const auto physical = [&](const std::vector<Access>& accesses) {
    std::vector<Range> ranges;
    ranges.reserve(accesses.size());
    for (const Access& access : accesses) {
      const uint32_t lo = resolve(access.at);
      ranges.push_back({.mem = access.mem, .lo = lo, .hi = lo + access.words});
    }
    return ranges;
  };
  WaitBits waits;
  for (size_t i = 0; i < plan.order.size(); ++i) {
    const Node& node = nodes_[plan.order[i]];
    std::vector<Range> reads = physical(node.reads);
    std::vector<Range> writes = physical(node.writes);
    const uint32_t wait =
        waits.Admit(static_cast<int32_t>(i), node.engine, reads, writes);
    done.instructions.push_back(Encode(node, resolve, wait));
    done.engines.push_back(node.engine);
    done.tags.push_back(node.tag);
    done.notes.push_back(node.note);
    done.reads.push_back(std::move(reads));
    done.writes.push_back(std::move(writes));
  }
  done.waits = waits.Count();
  done_ = std::move(done);
}

const Program::Done& Program::Finished() const {
  if (!done_.has_value()) {
    Fail("the program is not finalized");
  }
  return *done_;
}

size_t Program::WaitCount() const { return Finished().waits; }

size_t Program::EngineCount(Engine engine) const {
  return static_cast<size_t>(std::ranges::count_if(
      nodes_, [engine](const Node& node) { return node.engine == engine; }));
}

const std::vector<Instruction>& Program::Instructions() const {
  return Finished().instructions;
}

Engine Program::EngineOf(size_t index) const {
  return Finished().engines.at(index);
}

std::string_view Program::Tag(size_t index) const {
  return tag_names_.at(Finished().tags.at(index));
}

std::string_view Program::Note(size_t index) const {
  return Finished().notes.at(index);
}

const std::vector<Range>& Program::Reads(size_t index) const {
  return Finished().reads.at(index);
}

const std::vector<Range>& Program::Writes(size_t index) const {
  return Finished().writes.at(index);
}

uint32_t Program::Address(const Tile& tile) const {
  const Done& done = Finished();
  Require(tile.id_ < done.base.size() && done.base[tile.id_] != kNoTile,
          "the tile is not placed");
  return done.base[tile.id_];
}

void Program::PushScope(std::string_view tag) {
  const size_t length = scope_.size();
  if (!scope_.empty()) {
    scope_ += '/';
  }
  scope_ += tag;
  scope_lengths_.push_back(length);
  if (!tag_lookup_.contains(scope_)) {
    tag_lookup_.emplace(scope_, static_cast<uint32_t>(tag_names_.size()));
    tag_names_.push_back(scope_);
  }
}

void Program::PopScope() {
  Require(!scope_lengths_.empty(), "a scope is closed that was not opened");
  scope_.resize(scope_lengths_.back());
  scope_lengths_.pop_back();
}

std::vector<std::byte> Program::Bytes() const {
  return Serialize(Instructions());
}

std::string Program::Listing() const {
  const Done& done = Finished();
  std::string text;
  for (size_t i = 0; i < done.instructions.size(); ++i) {
    text += std::format("{:5d} {:4s} w={:x} {} {}\n", i,
                        kEngineNames.at(static_cast<size_t>(done.engines[i])),
                        done.instructions[i].Get(isa::kWait),
                        done.instructions[i].Hex(), done.notes[i]);
  }
  return text;
}

std::string Program::ListingJson() const {
  const Done& done = Finished();
  const auto ranges = [](const std::vector<Range>& list) {
    std::string text = "[";
    for (size_t i = 0; i < list.size(); ++i) {
      text += std::format("{}[\"{}\",{},{}]", i == 0 ? "" : ",",
                          MemName(list[i].mem), list[i].lo, list[i].hi);
    }
    return text + "]";
  };
  std::string text = std::format(R"({{"base":{},"tags":[)", base_);
  for (size_t i = 0; i < tag_names_.size(); ++i) {
    text += (i == 0 ? "" : ",") + Json(tag_names_[i]).Dump();
  }
  text += "],\"instructions\":[\n";
  for (size_t i = 0; i < done.instructions.size(); ++i) {
    const Instruction& instruction = done.instructions[i];
    text += std::format(
        "{}{{\"index\":{},\"engine\":\"{}\",\"opcode\":{},\"wait\":{},"
        "\"hex\":\"{}\",\"tag\":{},\"note\":{},\"reads\":{},"
        "\"writes\":{}}}",
        i == 0 ? "" : ",\n", i,
        kEngineNames.at(static_cast<size_t>(done.engines[i])),
        instruction.Get(isa::kOpcode), instruction.Get(isa::kWait),
        instruction.Hex(), done.tags[i], Json(done.notes[i]).Dump(),
        ranges(done.reads[i]), ranges(done.writes[i]));
  }
  return text + "\n]}\n";
}

}  // namespace hotarubi::compiler
