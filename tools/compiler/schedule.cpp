#include "schedule.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "hotarubi/isa.hpp"
#include "hotarubi/require.hpp"
#include "isa.hpp"
#include "timing.hpp"

namespace hotarubi::compiler {
namespace {

constexpr uint32_t kNone = kNoTile;
constexpr int kEngines = static_cast<int>(isa::kEngines);
constexpr int kCtrl = kEngines;
constexpr int kWaitHost = kEngines + 1;
constexpr int kVec = 2;

size_t MemIndex(Mem mem) { return static_cast<size_t>(mem); }

bool Intersect(uint32_t lo1, uint32_t hi1, uint32_t lo2, uint32_t hi2) {
  return lo1 < hi2 && lo2 < hi1;
}

class Graph {
 public:
  explicit Graph(size_t count)
      : successors_(count), predecessors_(count, 0), mark_(count, kNone) {}

  void Begin(uint32_t node) { node_ = node; }

  void From(uint32_t source) {
    if (source == kNone || source == node_ || mark_[source] == node_) {
      return;
    }
    mark_[source] = node_;
    successors_[source].push_back(node_);
    ++predecessors_[node_];
  }

  [[nodiscard]] const std::vector<uint32_t>& Successors(uint32_t node) const {
    return successors_[node];
  }
  [[nodiscard]] const std::vector<uint32_t>& Predecessors() const {
    return predecessors_;
  }

 private:
  std::vector<std::vector<uint32_t>> successors_;
  std::vector<uint32_t> predecessors_;
  std::vector<uint32_t> mark_;
  uint32_t node_ = kNone;
};

class Tracker {
 public:
  void Touch(uint64_t lo, uint64_t hi, bool write, Graph& graph,
             uint32_t node) {
    if (lo >= hi) {
      return;
    }
    Split(lo);
    Split(hi);
    auto it = segments_.lower_bound(lo);
    uint64_t at = lo;
    while (at < hi) {
      if (it == segments_.end() || it->first > at) {
        const uint64_t end =
            it == segments_.end() ? hi : std::min(hi, it->first);
        it = segments_.emplace_hint(it, at, Segment{.hi = end, .readers = {}});
      }
      Segment& segment = it->second;
      graph.From(segment.writer);
      if (write) {
        for (const uint32_t reader : segment.readers) {
          graph.From(reader);
        }
      } else if (segment.readers.empty() || segment.readers.back() != node) {
        segment.readers.push_back(node);
      }
      at = segment.hi;
      ++it;
    }
    if (write) {
      segments_.erase(segments_.lower_bound(lo), segments_.lower_bound(hi));
      segments_.emplace(lo, Segment{.hi = hi, .writer = node, .readers = {}});
    }
  }

 private:
  struct Segment {
    uint64_t hi = 0;
    uint32_t writer = kNone;
    std::vector<uint32_t> readers;
  };

  void Split(uint64_t at) {
    auto it = segments_.upper_bound(at);
    if (it == segments_.begin()) {
      return;
    }
    --it;
    if (it->first < at && at < it->second.hi) {
      Segment tail = it->second;
      it->second.hi = at;
      segments_.emplace(at, std::move(tail));
    }
  }

  std::map<uint64_t, Segment> segments_;
};

class Intervals {
 public:
  void Add(uint64_t lo, uint64_t hi) {
    auto it = spans_.upper_bound(lo);
    if (it != spans_.begin() && std::prev(it)->second >= lo) {
      --it;
      lo = it->first;
      hi = std::max(hi, it->second);
      it = spans_.erase(it);
    }
    while (it != spans_.end() && it->first <= hi) {
      hi = std::max(hi, it->second);
      it = spans_.erase(it);
    }
    spans_.emplace(lo, hi);
  }

  [[nodiscard]] bool Intersects(uint64_t lo, uint64_t hi) const {
    auto it = spans_.upper_bound(lo);
    if (it != spans_.begin() && std::prev(it)->second > lo) {
      return true;
    }
    return it != spans_.end() && it->first < hi;
  }

 private:
  std::map<uint64_t, uint64_t> spans_;
};

class Edges {
 public:
  Edges(const std::vector<Step>& steps, size_t tiles)
      : steps_(steps), graph_(steps.size()), tiles_(tiles) {
    for (const Step& step : steps) {
      if (step.dma && step.store && !step.extent.whole) {
        stored_.Add(step.extent.lo, step.extent.hi);
      }
    }
  }

  Graph Build() {
    for (uint32_t i = 0; i < steps_.size(); ++i) {
      graph_.Begin(i);
      graph_.From(last_barrier_);
      const Step& step = steps_[i];
      if (step.engine >= kCtrl) {
        for (const uint32_t node : since_barrier_) {
          graph_.From(node);
        }
        since_barrier_.clear();
        last_barrier_ = i;
        continue;
      }
      since_barrier_.push_back(i);
      OnChip(step, i);
      if (step.dma) {
        Ddr(step, i);
      }
      if (step.engine == kVec) {
        Vec(step, i);
      }
    }
    return std::move(graph_);
  }

 private:
  void OnChip(const Step& step, uint32_t node) {
    for (const bool write : {false, true}) {
      for (const Span& span : step.spans) {
        if (span.write != write) {
          continue;
        }
        Tracker& tracker = span.tile == kNone ? fixed_.at(MemIndex(span.mem))
                                              : tiles_.at(span.tile);
        tracker.Touch(span.lo, span.hi, write, graph_, node);
      }
    }
  }

  void Ddr(const Step& step, uint32_t node) {
    if (step.extent.whole) {
      graph_.From(last_whole_);
      for (const uint32_t store : stores_) {
        graph_.From(store);
      }
      stores_.clear();
      last_whole_ = node;
    } else if (step.store) {
      graph_.From(last_whole_);
      ddr_.Touch(step.extent.lo, step.extent.hi, true, graph_, node);
      stores_.push_back(node);
    } else if (stored_.Intersects(step.extent.lo, step.extent.hi)) {
      ddr_.Touch(step.extent.lo, step.extent.hi, false, graph_, node);
    }
  }

  void Vec(const Step& step, uint32_t node) {
    graph_.From(last_loaded_);
    if (step.loaded) {
      for (const uint32_t vec : vecs_) {
        graph_.From(vec);
      }
      vecs_.clear();
      last_loaded_ = node;
    } else {
      vecs_.push_back(node);
    }
  }

  const std::vector<Step>& steps_;
  Graph graph_;
  std::vector<Tracker> tiles_;
  std::array<Tracker, kMemCount> fixed_;
  Tracker ddr_;
  Intervals stored_;
  uint32_t last_whole_ = kNone;
  std::vector<uint32_t> stores_;
  uint32_t last_loaded_ = kNone;
  std::vector<uint32_t> vecs_;
  uint32_t last_barrier_ = kNone;
  std::vector<uint32_t> since_barrier_;
};

class Fenwick {
 public:
  explicit Fenwick(size_t size) : tree_(size + 1, 0) {}

  void Add(size_t index, int64_t value) {
    for (size_t i = index + 1; i < tree_.size(); i += i & (~i + 1)) {
      tree_[i] += value;
    }
  }

  [[nodiscard]] int64_t Prefix(size_t end) const {
    int64_t sum = 0;
    for (size_t i = end; i > 0; i -= i & (~i + 1)) {
      sum += tree_[i];
    }
    return sum;
  }

 private:
  std::vector<int64_t> tree_;
};

struct Physical {
  Mem mem = Mem::kFbuf;
  uint32_t lo = 0;
  uint32_t hi = 0;
  bool write = false;
};

constexpr uint32_t kHorizon = 256;

using Holes = std::vector<std::pair<uint32_t, uint32_t>>;

class Scheduler {
 public:
  Scheduler(const std::vector<Step>& steps, const std::vector<TileSize>& tiles,
            const ScheduleOptions& options)
      : steps_(steps),
        tiles_(tiles),
        window_(std::max(options.window, 1U)),
        graph_(Edges(steps, tiles.size()).Build()),
        waiting_(graph_.Predecessors()),
        scheduled_(steps.size(), false),
        last_issue_(options.timing.start_cycles),
        base_(tiles.size(), kNone),
        left_(tiles.size(), 0),
        owner_(tiles.size(), kNone),
        last_(tiles.size(), 0),
        node_tiles_(steps.size()) {
    pending_.fill(kNone);
    for (size_t m = 0; m < kMemCount; ++m) {
      reserve_.emplace_back(steps.size());
      holes_.at(m).emplace(0, kMemDepth.at(m));
    }
    for (uint32_t node = 0; node < steps.size(); ++node) {
      std::vector<uint32_t>& list = node_tiles_[node];
      for (const Span& span : steps[node].spans) {
        if (span.tile != kNone &&
            std::ranges::find(list, span.tile) == list.end()) {
          list.push_back(span.tile);
        }
      }
      for (const uint32_t tile : list) {
        ++left_[tile];
        last_[tile] = node;
        if (owner_[tile] == kNone) {
          owner_[tile] = node;
          reserve_.at(MemIndex(tiles[tile].mem)).Add(node, tiles[tile].words);
        }
      }
    }
  }

  Plan Run() {
    Admit();
    while (order_.size() < steps_.size()) {
      std::vector<uint32_t> blocked;
      for (;;) {
        const uint32_t best = Choose(blocked);
        Require(best != kNone,
                std::format("no instruction can be scheduled after {} of {} "
                            "instructions; the oldest is {}",
                            order_.size(), steps_.size(), oldest_));
        if (Commit(best)) {
          break;
        }
        blocked.push_back(best);
      }
    }
    return {.order = std::move(order_), .base = std::move(base_)};
  }

 private:
  struct Issue {
    uint64_t issue = 0;
    uint64_t completion = 0;
    uint32_t need = 0;
  };

  void Admit() {
    const uint64_t limit = uint64_t{oldest_} + window_;
    while (admitted_ < steps_.size() && admitted_ < limit) {
      if (waiting_[admitted_] == 0) {
        ready_.insert(admitted_);
      }
      ++admitted_;
    }
  }

  [[nodiscard]] uint32_t Choose(const std::vector<uint32_t>& blocked) const {
    const uint64_t earliest = first_ ? last_issue_ : last_issue_ + 1;
    uint32_t best = kNone;
    Issue best_issue;
    for (const uint32_t node : ready_) {
      if (std::ranges::find(blocked, node) != blocked.end() ||
          !Reserved(node)) {
        continue;
      }
      const Issue issue = Evaluate(node);
      if (best == kNone || issue.issue < best_issue.issue) {
        best = node;
        best_issue = issue;
      }
      if (issue.issue == earliest) {
        break;
      }
    }
    return best;
  }

  [[nodiscard]] bool Reserved(uint32_t node) const {
    std::array<uint64_t, kMemCount> need{};
    std::array<uint64_t, kMemCount> older{};
    for (const uint32_t tile : node_tiles_[node]) {
      if (base_[tile] != kNone) {
        continue;
      }
      const size_t m = MemIndex(tiles_[tile].mem);
      need.at(m) += tiles_[tile].words;
      if (owner_[tile] < node) {
        older.at(m) += tiles_[tile].words;
      }
    }
    for (size_t m = 0; m < kMemCount; ++m) {
      if (need.at(m) == 0) {
        continue;
      }
      const auto reserved =
          static_cast<uint64_t>(reserve_.at(m).Prefix(node)) - older.at(m);
      if (live_.at(m) + need.at(m) + reserved > kMemDepth.at(m)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] Issue Evaluate(uint32_t node) const {
    const Step& step = steps_[node];
    Issue result;
    result.issue = first_ ? last_issue_ : last_issue_ + 1;
    if (step.engine == kWaitHost) {
      result.completion = result.issue;
      return result;
    }
    if (step.engine == kCtrl) {
      result.need = (1U << kEngines) - 1;
      for (int e = 0; e < kEngines; ++e) {
        result.issue = std::max(result.issue, free_.at(e) + 1);
      }
      result.completion = result.issue;
      return result;
    }
    result.issue = std::max(result.issue, free_.at(step.engine) + 1);
    for (int e = 0; e < kEngines; ++e) {
      if (e != step.engine && pending_.at(e) != kNone && Overlaps(node, e)) {
        result.need |= 1U << e;
        result.issue = std::max(result.issue, free_.at(e) + 1);
      }
    }
    result.completion = result.issue + step.cost;
    return result;
  }

  [[nodiscard]] bool Overlaps(uint32_t node, int engine) const {
    for (const Span& span : steps_[node].spans) {
      uint32_t lo = span.lo;
      uint32_t hi = span.hi;
      if (span.tile != kNone) {
        if (base_[span.tile] == kNone) {
          continue;
        }
        lo += base_[span.tile];
        hi += base_[span.tile];
      }
      for (const Physical& other : pending_spans_.at(engine)) {
        if (other.mem == span.mem && (other.write || span.write) &&
            Intersect(lo, hi, other.lo, other.hi)) {
          return true;
        }
      }
    }
    return false;
  }

  [[nodiscard]] std::vector<Physical> PhysicalSpans(uint32_t node) const {
    std::vector<Physical> spans;
    for (const Span& span : steps_[node].spans) {
      const uint32_t base = span.tile == kNone ? 0 : base_[span.tile];
      spans.push_back({.mem = span.mem,
                       .lo = base + span.lo,
                       .hi = base + span.hi,
                       .write = span.write});
    }
    return spans;
  }

  bool Commit(uint32_t node) {
    if (!PlaceTiles(node)) {
      return false;
    }
    const Issue issue = Evaluate(node);
    last_issue_ = issue.issue;
    first_ = false;
    Occupy(node, issue);
    for (const uint32_t tile : node_tiles_[node]) {
      if (--left_[tile] == 0) {
        Free(tile);
      }
    }
    scheduled_[node] = true;
    ready_.erase(node);
    order_.push_back(node);
    for (const uint32_t next : graph_.Successors(node)) {
      if (--waiting_[next] == 0 && next < admitted_) {
        ready_.insert(next);
      }
    }
    while (oldest_ < steps_.size() && scheduled_[oldest_]) {
      ++oldest_;
    }
    Admit();
    return true;
  }

  bool PlaceTiles(uint32_t node) {
    std::vector<uint32_t> unplaced;
    for (const uint32_t tile : node_tiles_[node]) {
      if (base_[tile] == kNone) {
        unplaced.push_back(tile);
      }
    }
    std::ranges::stable_sort(unplaced, std::greater<>(),
                             [&](uint32_t tile) { return tiles_[tile].words; });
    for (size_t i = 0; i < unplaced.size(); ++i) {
      if (!Place(unplaced[i], steps_[node].engine, node == oldest_)) {
        for (size_t j = 0; j < i; ++j) {
          Release(unplaced[j]);
        }
        return false;
      }
    }
    return true;
  }

  void Occupy(uint32_t node, const Issue& issue) {
    const int engine = steps_[node].engine;
    if (engine == kCtrl) {
      pending_.fill(kNone);
      for (auto& spans : pending_spans_) {
        spans.clear();
      }
      return;
    }
    if (engine >= kEngines) {
      return;
    }
    for (int e = 0; e < kEngines; ++e) {
      if ((issue.need & (1U << e)) != 0) {
        pending_.at(e) = kNone;
        pending_spans_.at(e).clear();
      }
    }
    pending_.at(engine) = node;
    pending_spans_.at(engine) = PhysicalSpans(node);
    free_.at(engine) = issue.completion;
  }

  bool Place(uint32_t tile, int engine, bool forced) {
    const size_t m = MemIndex(tiles_[tile].mem);
    const uint32_t words = tiles_[tile].words;
    const uint32_t death = last_[tile];
    std::vector<std::pair<uint32_t, uint32_t>> dirty;
    for (int e = 0; e < kEngines; ++e) {
      if (e == engine) {
        continue;
      }
      for (const Physical& span : pending_spans_.at(e)) {
        if (span.mem == tiles_[tile].mem) {
          dirty.emplace_back(span.lo, span.hi);
        }
      }
    }
    struct Choice {
      bool clean = false;
      bool outlived = false;
      uint32_t gap = 0;
      uint32_t size = 0;
      uint32_t lo = 0;

      [[nodiscard]] auto Rank() const {
        return std::tuple{!clean, !outlived, gap, size, lo};
      }
    };
    std::vector<Choice> choices;
    const auto consider = [&](uint32_t lo, uint32_t size, uint32_t neighbor) {
      choices.push_back(
          {.clean = std::ranges::none_of(
               dirty,
               [&](const auto& span) {
                 return Intersect(lo, lo + words, span.first, span.second);
               }),
           .outlived = neighbor >= death,
           .gap = neighbor >= death ? neighbor - death : death - neighbor,
           .size = size,
           .lo = lo});
    };
    for (const auto& [lo, hi] : holes_.at(m)) {
      if (hi - lo < words) {
        continue;
      }
      consider(lo, hi - lo, Below(m, lo));
      if (hi - words != lo) {
        consider(hi - words, hi - lo, Above(m, hi));
      }
    }
    if (choices.empty()) {
      return false;
    }
    std::ranges::sort(choices, {}, &Choice::Rank);
    for (const Choice& choice : choices) {
      Take(tile, m, choice.lo);
      if (Safe()) {
        return true;
      }
      Release(tile);
    }
    if (!forced) {
      return false;
    }
    Take(tile, m, choices.front().lo);
    return true;
  }

  void Take(uint32_t tile, size_t m, uint32_t lo) {
    const uint32_t words = tiles_[tile].words;
    Cut(m, lo, words);
    base_[tile] = lo;
    occupants_.at(m).emplace(lo, tile);
    live_.at(m) += words;
    reserve_.at(m).Add(owner_[tile], -static_cast<int64_t>(words));
  }

  [[nodiscard]] bool Safe() const {
    const auto end = static_cast<uint32_t>(
        std::min<uint64_t>(steps_.size(), uint64_t{oldest_} + kHorizon));
    std::array<uint64_t, kMemCount> wanted{};
    std::vector<uint32_t> seen;
    for (uint32_t node = oldest_; node < end; ++node) {
      if (scheduled_[node]) {
        continue;
      }
      for (const uint32_t tile : node_tiles_[node]) {
        if (base_[tile] == kNone &&
            std::ranges::find(seen, tile) == seen.end()) {
          seen.push_back(tile);
          wanted.at(MemIndex(tiles_[tile].mem)) += tiles_[tile].words;
        }
      }
    }
    bool roomy = true;
    for (size_t m = 0; m < kMemCount; ++m) {
      uint32_t largest = 0;
      for (const auto& [lo, hi] : holes_.at(m)) {
        largest = std::max(largest, hi - lo);
      }
      roomy = roomy && wanted.at(m) <= largest;
    }
    return roomy || Simulate(end);
  }

  [[nodiscard]] bool Simulate(uint32_t end) const {
    std::array<Holes, kMemCount> holes;
    for (size_t m = 0; m < kMemCount; ++m) {
      holes.at(m).assign(holes_.at(m).begin(), holes_.at(m).end());
    }
    std::map<uint32_t, uint32_t> placed;
    std::map<uint32_t, uint32_t> left;
    for (uint32_t node = oldest_; node < end; ++node) {
      if (scheduled_[node]) {
        continue;
      }
      for (const uint32_t tile : node_tiles_[node]) {
        if (base_[tile] != kNone || placed.contains(tile)) {
          continue;
        }
        const auto lo =
            BestFit(holes.at(MemIndex(tiles_[tile].mem)), tiles_[tile].words);
        if (!lo) {
          return false;
        }
        placed.emplace(tile, *lo);
      }
      for (const uint32_t tile : node_tiles_[node]) {
        const auto [it, fresh] = left.try_emplace(tile, left_[tile]);
        if (--it->second == 0) {
          const auto at = placed.find(tile);
          const uint32_t lo = at != placed.end() ? at->second : base_[tile];
          Return(holes.at(MemIndex(tiles_[tile].mem)), lo,
                 lo + tiles_[tile].words);
        }
      }
    }
    return true;
  }

  static std::optional<uint32_t> BestFit(Holes& holes, uint32_t words) {
    auto best = holes.end();
    for (auto it = holes.begin(); it != holes.end(); ++it) {
      if (it->second - it->first >= words &&
          (best == holes.end() ||
           it->second - it->first < best->second - best->first)) {
        best = it;
      }
    }
    if (best == holes.end()) {
      return std::nullopt;
    }
    const uint32_t lo = best->first;
    best->first += words;
    if (best->first == best->second) {
      holes.erase(best);
    }
    return lo;
  }

  static void Return(Holes& holes, uint32_t lo, uint32_t hi) {
    holes.emplace_back(lo, hi);
    std::ranges::sort(holes);
    Holes merged;
    for (const auto& hole : holes) {
      if (!merged.empty() && merged.back().second == hole.first) {
        merged.back().second = hole.second;
      } else {
        merged.push_back(hole);
      }
    }
    holes = std::move(merged);
  }

  [[nodiscard]] uint32_t Below(size_t m, uint32_t lo) const {
    if (lo == 0) {
      return kNone;
    }
    const auto& occupants = occupants_.at(m);
    return last_[std::prev(occupants.upper_bound(lo - 1))->second];
  }

  [[nodiscard]] uint32_t Above(size_t m, uint32_t hi) const {
    if (hi == kMemDepth.at(m)) {
      return kNone;
    }
    return last_[occupants_.at(m).at(hi)];
  }

  void Cut(size_t m, uint32_t lo, uint32_t words) {
    auto& holes = holes_.at(m);
    auto it = std::prev(holes.upper_bound(lo));
    const uint32_t hole_lo = it->first;
    const uint32_t hole_hi = it->second;
    holes.erase(it);
    if (hole_lo < lo) {
      holes.emplace(hole_lo, lo);
    }
    if (lo + words < hole_hi) {
      holes.emplace(lo + words, hole_hi);
    }
  }

  void Give(size_t m, uint32_t lo, uint32_t hi) {
    auto& holes = holes_.at(m);
    auto it = holes.upper_bound(lo);
    if (it != holes.end() && it->first == hi) {
      hi = it->second;
      it = holes.erase(it);
    }
    if (it != holes.begin() && std::prev(it)->second == lo) {
      --it;
      lo = it->first;
      holes.erase(it);
    }
    holes.emplace(lo, hi);
  }

  void Free(uint32_t tile) {
    const size_t m = MemIndex(tiles_[tile].mem);
    Give(m, base_[tile], base_[tile] + tiles_[tile].words);
    occupants_.at(m).erase(base_[tile]);
    live_.at(m) -= tiles_[tile].words;
  }

  void Release(uint32_t tile) {
    Free(tile);
    reserve_.at(MemIndex(tiles_[tile].mem))
        .Add(owner_[tile], tiles_[tile].words);
    base_[tile] = kNone;
  }

  const std::vector<Step>& steps_;
  const std::vector<TileSize>& tiles_;
  uint32_t window_;
  Graph graph_;
  std::vector<uint32_t> waiting_;
  std::vector<bool> scheduled_;
  std::set<uint32_t> ready_;
  uint32_t oldest_ = 0;
  uint32_t admitted_ = 0;
  uint64_t last_issue_;
  bool first_ = true;
  std::array<uint64_t, kEngines> free_{};
  std::array<uint32_t, kEngines> pending_{};
  std::array<std::vector<Physical>, kEngines> pending_spans_;
  std::vector<uint32_t> base_;
  std::vector<uint32_t> left_;
  std::vector<uint32_t> owner_;
  std::vector<uint32_t> last_;
  std::vector<std::vector<uint32_t>> node_tiles_;
  std::array<std::map<uint32_t, uint32_t>, kMemCount> holes_;
  std::array<std::map<uint32_t, uint32_t>, kMemCount> occupants_;
  std::array<uint64_t, kMemCount> live_{};
  std::vector<Fenwick> reserve_;
  std::vector<uint32_t> order_;
};

}  // namespace

Plan ScheduleStream(const std::vector<Step>& steps,
                    const std::vector<TileSize>& tiles,
                    const ScheduleOptions& options) {
  for (const TileSize& tile : tiles) {
    Require(
        tile.words >= 1 && tile.words <= kMemDepth.at(MemIndex(tile.mem)),
        std::format("a tile of {} words does not fit its memory", tile.words));
  }
  return Scheduler(steps, tiles, options).Run();
}

}  // namespace hotarubi::compiler
