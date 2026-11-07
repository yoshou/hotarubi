#include "schedule.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

#include "isa.hpp"
#include "machine.hpp"
#include "program.hpp"
#include "timing.hpp"
#include "word.hpp"

namespace hotarubi::compiler {
namespace {

constexpr int kEngines = 3;
constexpr uint32_t kOpDmaLoad = 1;
constexpr uint32_t kOpDmaStore = 2;
constexpr uint32_t kOpVec = 5;
constexpr uint32_t kModeGather = 3;
constexpr uint32_t kMemTbuf = 4;
constexpr uint32_t kMemCbuf = 5;

struct Node {
  int engine = 0;
  bool barrier = false;
  bool dma = false;
  bool store = false;
  bool gather = false;
  bool run = false;
  bool table_load = false;
  iss::DdrExtent extent;
  uint64_t cost = 0;
};

iss::Word128 Word(const Instruction& instruction) {
  const auto& w = instruction.words;
  return {.low = w[0] | (uint64_t{w[1]} << 32),
          .high = w[2] | (uint64_t{w[3]} << 32)};
}

class Estimator {
 public:
  explicit Estimator(const iss::TimingParameters& timing) : model_(timing) {}

  uint64_t Cycles(const iss::Word128& word) {
    if (word.Field(120, 8) != kOpVec) {
      return iss::EstimateCycles(word, model_);
    }
    const iss::VecCommand command = iss::DecodeVec(word);
    const auto key = std::make_tuple(command.op, command.quantize,
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

Node Describe(const Program& program, size_t index, Estimator* estimator) {
  const Engine engine = program.EngineOf(index);
  const iss::Word128 word = Word(program.Instructions()[index]);
  Node node;
  node.engine = static_cast<int>(engine);
  node.barrier = engine == Engine::kCtrl || engine == Engine::kWait;
  const uint32_t opcode = word.Field(120, 8);
  if (opcode == kOpDmaLoad || opcode == kOpDmaStore) {
    const iss::DmaCommand command = iss::DecodeDma(word);
    node.dma = true;
    node.store = command.store;
    node.gather = !command.store && command.mode == kModeGather;
    node.table_load =
        !command.store && (command.mem == kMemTbuf || command.mem == kMemCbuf);
    node.extent = iss::EstimateDdrExtent(command);
  } else if (opcode == kOpVec) {
    node.run = iss::DecodeVec(word).op >= iss::kVecRun;
  }
  if (estimator != nullptr && !node.barrier) {
    node.cost = std::max<uint64_t>(estimator->Cycles(word), 1);
  }
  return node;
}

bool Overlap(const Range& a, const Range& b) {
  return a.mem == b.mem && a.lo < b.hi && b.lo < a.hi;
}

bool Overlap(const std::vector<Range>& a, const std::vector<Range>& b) {
  return std::ranges::any_of(a, [&](const Range& x) {
    return std::ranges::any_of(b,
                               [&](const Range& y) { return Overlap(x, y); });
  });
}

bool Overlap(const iss::DdrExtent& a, const iss::DdrExtent& b) {
  if (a.whole || b.whole) {
    return true;
  }
  return a.lo < b.hi && b.lo < a.hi;
}

bool ConflictsWith(const Program& program, const Node& earlier, size_t j,
                   const Node& later, size_t i) {
  if (earlier.barrier || later.barrier) {
    return true;
  }
  const auto& reads_j = program.Reads(j);
  const auto& writes_j = program.Writes(j);
  const auto& reads_i = program.Reads(i);
  const auto& writes_i = program.Writes(i);
  if (Overlap(writes_j, reads_i) || Overlap(writes_j, writes_i) ||
      Overlap(reads_j, writes_i)) {
    return true;
  }
  if (earlier.dma && later.dma) {
    if ((earlier.store || later.store) &&
        Overlap(earlier.extent, later.extent)) {
      return true;
    }
    if (earlier.gather && later.gather) {
      return true;
    }
  }
  return earlier.engine == static_cast<int>(Engine::kVec) &&
         later.engine == static_cast<int>(Engine::kVec) &&
         (earlier.run || later.run);
}

class Scheduler {
 public:
  Scheduler(const Program& program, const ScheduleOptions& options)
      : program_(program),
        options_(options),
        estimator_(options.timing),
        count_(program.Size()),
        nodes_(count_),
        described_(count_, false),
        scheduled_(count_, false),
        predecessors_(count_, 0),
        successors_(count_) {}

  std::vector<size_t> Run() {
    std::vector<size_t> order;
    order.reserve(count_);
    last_issue_ = options_.timing.start_cycles;
    bool first = true;
    while (order.size() < count_) {
      Fill();
      const size_t chosen = Choose(first);
      first = false;
      Commit(chosen);
      order.push_back(chosen);
    }
    return order;
  }

 private:
  struct Issue {
    uint64_t issue = 0;
    uint64_t completion = 0;
    uint32_t need = 0;
  };

  const Node& NodeOf(size_t index) {
    if (!described_[index]) {
      nodes_[index] = Describe(program_, index, &estimator_);
      described_[index] = true;
    }
    return nodes_[index];
  }

  void Fill() {
    while (next_ < count_ && window_.size() < options_.window) {
      const size_t index = next_++;
      const Node& node = NodeOf(index);
      for (const size_t earlier : window_) {
        if (!scheduled_[earlier] &&
            ConflictsWith(program_, NodeOf(earlier), earlier, node, index)) {
          successors_[earlier].push_back(index);
          ++predecessors_[index];
        }
      }
      window_.push_back(index);
    }
  }

  Issue Evaluate(size_t index, bool first) {
    const Node& node = NodeOf(index);
    Issue result;
    result.issue = first ? last_issue_ : last_issue_ + 1;
    if (node.barrier) {
      if (program_.EngineOf(index) == Engine::kCtrl) {
        result.need = (1U << kEngines) - 1;
      }
    } else {
      result.issue = std::max(result.issue, free_.at(node.engine) + 1);
      for (int e = 0; e < kEngines; ++e) {
        if (e == node.engine) {
          continue;
        }
        const bool conflict =
            std::ranges::any_of(pending_.at(e), [&](size_t p) {
              return ConflictsWith(program_, NodeOf(p), p, node, index);
            });
        if (conflict) {
          result.need |= 1U << e;
        }
      }
    }
    for (int e = 0; e < kEngines; ++e) {
      if ((result.need & (1U << e)) != 0) {
        result.issue = std::max(result.issue, free_.at(e) + 1);
      }
    }
    uint64_t start = result.issue;
    if (node.table_load) {
      start = std::max(start, free_.at(static_cast<int>(Engine::kVec)));
    }
    result.completion = node.barrier ? result.issue : start + node.cost;
    return result;
  }

  size_t Choose(bool first) {
    size_t best = count_;
    Issue best_issue;
    for (const size_t index : window_) {
      if (scheduled_[index] || predecessors_[index] != 0) {
        continue;
      }
      const Issue issue = Evaluate(index, first);
      if (best == count_ || issue.issue < best_issue.issue) {
        best = index;
        best_issue = issue;
      }
      if (issue.issue == (first ? last_issue_ : last_issue_ + 1)) {
        break;
      }
    }
    Require(best < count_, "the scheduler found no ready instruction");
    chosen_ = best_issue;
    return best;
  }

  void Commit(size_t index) {
    const Node& node = NodeOf(index);
    scheduled_[index] = true;
    for (const size_t later : successors_[index]) {
      --predecessors_[later];
    }
    for (int e = 0; e < kEngines; ++e) {
      if ((chosen_.need & (1U << e)) != 0) {
        pending_.at(e).clear();
      }
    }
    if (!node.barrier) {
      pending_.at(node.engine).clear();
      pending_.at(node.engine).push_back(index);
      free_.at(node.engine) = chosen_.completion;
    }
    last_issue_ = chosen_.issue;
    while (!window_.empty() && scheduled_[window_.front()]) {
      window_.pop_front();
    }
  }

  const Program& program_;
  ScheduleOptions options_;
  Estimator estimator_;
  size_t count_;
  std::vector<Node> nodes_;
  std::vector<bool> described_;
  std::vector<bool> scheduled_;
  std::vector<uint32_t> predecessors_;
  std::vector<std::vector<size_t>> successors_;
  std::deque<size_t> window_;
  size_t next_ = 0;
  uint64_t last_issue_ = 0;
  std::array<uint64_t, kEngines> free_{};
  std::array<std::vector<size_t>, kEngines> pending_;
  Issue chosen_;
};

}  // namespace

bool Conflicts(const Program& program, size_t earlier, size_t later) {
  Require(earlier < later && later < program.Size(),
          "Conflicts needs two instructions in emission order");
  return ConflictsWith(program, Describe(program, earlier, nullptr), earlier,
                       Describe(program, later, nullptr), later);
}

std::vector<size_t> ScheduleOrder(const Program& program,
                                  const ScheduleOptions& options) {
  Require(options.window > 0, "the scheduling window must not be empty");
  return Scheduler(program, options).Run();
}

void Schedule(Program& program, const ScheduleOptions& options) {
  program.Reorder(ScheduleOrder(program, options));
}

}  // namespace hotarubi::compiler
