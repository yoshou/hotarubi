#include "report.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "analysis.hpp"
#include "hotarubi/isa.hpp"
#include "hotarubi/json.hpp"
#include "machine.hpp"
#include "timing.hpp"
#include "trace_record.hpp"

namespace hotarubi::trace {
namespace {

using isa::kOpDmaLoad;
using isa::kOpDmaStore;
using isa::kOpGemm;
using isa::kOpVec;

double Ms(uint64_t cycles, uint64_t clock_hz) {
  return static_cast<double>(cycles) * 1000.0 / static_cast<double>(clock_hz);
}

Json Number(uint64_t value) { return Json(static_cast<double>(value)); }

Json Milliseconds(uint64_t cycles, uint64_t clock_hz) {
  return Json(Ms(cycles, clock_hz));
}

std::vector<std::pair<std::string, const Aggregate*>> BySpan(
    const std::map<std::string, Aggregate>& aggregates) {
  std::vector<std::pair<std::string, const Aggregate*>> rows;
  rows.reserve(aggregates.size());
  for (const auto& [name, aggregate] : aggregates) {
    rows.emplace_back(name, &aggregate);
  }
  std::ranges::stable_sort(rows, [](const auto& a, const auto& b) {
    return a.second->span > b.second->span;
  });
  return rows;
}

std::string Percent(uint64_t part, uint64_t whole) {
  if (whole == 0) {
    return "-";
  }
  return std::format("{:5.1f}%", 100.0 * static_cast<double>(part) /
                                     static_cast<double>(whole));
}

std::string AggregateTable(const std::map<std::string, Aggregate>& aggregates,
                           const ReportOptions& options, bool statistics,
                           std::string_view label) {
  std::string text = std::format(
      "{:<28s} {:>8s} {:>10s} {:>10s} {:>10s} {:>6s} {:>10s}", label, "count",
      "span ms", "excl ms", "stream ms", "eff", "model ms");
  if (statistics) {
    text += std::format(" {:>12s} {:>12s} {:>12s} {:>8s}", "A", "B", "C", "D");
  }
  text += '\n';
  for (const auto& [name, aggregate] : BySpan(aggregates)) {
    text += std::format(
        "{:<28s} {:>8d} {:>10.3f} {:>10.3f} {:>10.3f} {:>6s} {:>10.3f}",
        name.empty() ? "(untagged)" : name.substr(0, 28), aggregate->count,
        Ms(aggregate->span, options.clock_hz),
        Ms(aggregate->exclusive, options.clock_hz),
        Ms(aggregate->streaming, options.clock_hz),
        Percent(aggregate->streaming, aggregate->span),
        Ms(aggregate->model, options.clock_hz));
    if (statistics) {
      text += std::format(" {:>12d} {:>12d} {:>12d} {:>8d}",
                          aggregate->statistics.a, aggregate->statistics.b,
                          aggregate->statistics.c, aggregate->statistics.d);
    }
    text += '\n';
  }
  return text;
}

std::string PhaseTable(const Analysis& analysis, const ReportOptions& options) {
  const std::map<std::string, Aggregate> tags = ByTag(analysis, options.depth);
  std::vector<std::pair<std::string, const Aggregate*>> rows;
  rows.reserve(tags.size());
  for (const auto& [name, aggregate] : tags) {
    rows.emplace_back(name, &aggregate);
  }
  std::ranges::stable_sort(rows, [](const auto& a, const auto& b) {
    return a.second->first < b.second->first;
  });
  std::string text =
      std::format("{:<28s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}\n",
                  std::format("phase (depth {})", options.depth), "start ms",
                  "span ms", "DMA ms", "GEMM ms", "VEC ms");
  for (const auto& [name, aggregate] : rows) {
    text += std::format(
        "{:<28s} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}\n",
        name.empty() ? "(untagged)" : name.substr(0, 28),
        Ms(aggregate->first, options.clock_hz),
        Ms(aggregate->Span(), options.clock_hz),
        Ms(aggregate->busy.at(0), options.clock_hz),
        Ms(aggregate->busy.at(1), options.clock_hz),
        Ms(aggregate->busy.at(2), options.clock_hz));
  }
  return text;
}

std::vector<const Event*> LargestGaps(const Analysis& analysis, size_t count) {
  std::vector<const Event*> gaps;
  for (const Event& event : analysis.events) {
    if (event.gap > 0) {
      gaps.push_back(&event);
    }
  }
  std::ranges::stable_sort(
      gaps, [](const Event* a, const Event* b) { return a->gap > b->gap; });
  if (gaps.size() > count) {
    gaps.resize(count);
  }
  return gaps;
}

Json AggregateJson(const std::map<std::string, Aggregate>& aggregates,
                   const ReportOptions& options) {
  Json::Object object;
  for (const auto& [name, aggregate] : BySpan(aggregates)) {
    Json::Object entry = {
        {"count", Number(aggregate->count)},
        {"span_ms", Milliseconds(aggregate->span, options.clock_hz)},
        {"exclusive_ms", Milliseconds(aggregate->exclusive, options.clock_hz)},
        {"streaming_ms", Milliseconds(aggregate->streaming, options.clock_hz)},
        {"model_ms", Milliseconds(aggregate->model, options.clock_hz)},
        {"start_ms", Milliseconds(aggregate->first, options.clock_hz)},
        {"phase_span_ms", Milliseconds(aggregate->Span(), options.clock_hz)},
        {"busy_ms",
         Json(Json::Array{
             Milliseconds(aggregate->busy.at(0), options.clock_hz),
             Milliseconds(aggregate->busy.at(1), options.clock_hz),
             Milliseconds(aggregate->busy.at(2), options.clock_hz)})}};
    if (aggregate->statistics_count > 0) {
      entry.emplace_back("statistics",
                         Json(Json::Array{Number(aggregate->statistics.a),
                                          Number(aggregate->statistics.b),
                                          Number(aggregate->statistics.c),
                                          Number(aggregate->statistics.d)}));
    }
    object.emplace_back(name, Json(std::move(entry)));
  }
  return Json(std::move(object));
}

struct LeastSquares {
  size_t unknowns = 0;
  std::vector<double> normal;
  std::vector<double> rhs;
  size_t rows = 0;

  explicit LeastSquares(size_t n)
      : unknowns(n), normal(n * n, 0.0), rhs(n, 0.0) {}

  void Add(const std::vector<double>& features, double target) {
    ++rows;
    for (size_t i = 0; i < unknowns; ++i) {
      rhs[i] += features[i] * target;
      for (size_t j = 0; j < unknowns; ++j) {
        normal[(i * unknowns) + j] += features[i] * features[j];
      }
    }
  }

  [[nodiscard]] std::vector<double> Solve() const {
    std::vector<double> a = normal;
    std::vector<double> b = rhs;
    const size_t n = unknowns;
    std::vector<double> x(n, 0.0);
    for (size_t column = 0; column < n; ++column) {
      size_t pivot = column;
      for (size_t row = column + 1; row < n; ++row) {
        if (std::fabs(a[(row * n) + column]) >
            std::fabs(a[(pivot * n) + column])) {
          pivot = row;
        }
      }
      if (std::fabs(a[(pivot * n) + column]) < 1e-9) {
        continue;
      }
      for (size_t j = 0; j < n; ++j) {
        std::swap(a[(column * n) + j], a[(pivot * n) + j]);
      }
      std::swap(b[column], b[pivot]);
      for (size_t row = 0; row < n; ++row) {
        if (row == column) {
          continue;
        }
        const double factor = a[(row * n) + column] / a[(column * n) + column];
        for (size_t j = 0; j < n; ++j) {
          a[(row * n) + j] -= factor * a[(column * n) + j];
        }
        b[row] -= factor * b[column];
      }
    }
    for (size_t i = 0; i < n; ++i) {
      x[i] = std::fabs(a[(i * n) + i]) < 1e-9 ? 0.0 : b[i] / a[(i * n) + i];
    }
    return x;
  }
};

uint32_t Round(double value) {
  return value <= 0.0 ? 0U : static_cast<uint32_t>(std::llround(value));
}

double Median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::ranges::sort(values);
  return values[values.size() / 2];
}

}  // namespace

std::string ReportText(const Analysis& analysis, const ReportOptions& options) {
  const uint64_t clock = options.clock_hz;
  std::string text = std::format(
      "stream {}: {} instructions, {} cycles, {:.3f} ms at {} Hz\n",
      analysis.stream, analysis.events.size(), analysis.total_cycles,
      Ms(analysis.total_cycles, clock), clock);
  text += std::format("first issue at cycle {}\n", analysis.first_issue);
  for (int e = 0; e < kEngines; ++e) {
    text += std::format("{:<5s} busy {:>10.3f} ms {}\n", kSourceNames.at(e),
                        Ms(analysis.busy.at(e), clock),
                        Percent(analysis.busy.at(e), analysis.total_cycles));
  }
  text += "\noccupancy\n";
  for (const auto& [mask, cycles] : analysis.occupancy) {
    text += std::format("{:<16s} {:>10.3f} ms {}\n", mask, Ms(cycles, clock),
                        Percent(cycles, analysis.total_cycles));
  }
  text += "\nissue gaps by blocker\n";
  for (const auto& [blocker, cycles] : analysis.blockers) {
    text += std::format("{:<16s} {:>10.3f} ms {}\n", blocker, Ms(cycles, clock),
                        Percent(cycles, analysis.total_cycles));
  }
  text += '\n' + PhaseTable(analysis, options);
  text += '\n' + AggregateTable(analysis.categories, options,
                                analysis.has_statistics, "category");
  text += "\nlargest issue gaps\n";
  for (const Event* event : LargestGaps(analysis, options.gaps)) {
    text += std::format("{:>8d} {:>10.3f} ms at {:>10.3f} ms {:<5s} {} {}\n",
                        event->index, Ms(event->gap, clock),
                        Ms(event->issue, clock), event->blocker,
                        event->category, event->note);
  }
  return text;
}

Json ReportJson(const Analysis& analysis, const ReportOptions& options) {
  const uint64_t clock = options.clock_hz;
  Json::Object occupancy;
  for (const auto& [mask, cycles] : analysis.occupancy) {
    occupancy.emplace_back(mask, Milliseconds(cycles, clock));
  }
  Json::Object blockers;
  for (const auto& [blocker, cycles] : analysis.blockers) {
    blockers.emplace_back(blocker, Milliseconds(cycles, clock));
  }
  Json::Array gaps;
  for (const Event* event : LargestGaps(analysis, options.gaps)) {
    gaps.emplace_back(
        Json::Object{{"index", Number(event->index)},
                     {"gap_ms", Milliseconds(event->gap, clock)},
                     {"issue_ms", Milliseconds(event->issue, clock)},
                     {"blocker", Json(event->blocker)},
                     {"category", Json(event->category)},
                     {"note", Json(event->note)}});
  }
  return Json(Json::Object{
      {"stream", Json(analysis.stream)},
      {"instructions", Number(analysis.events.size())},
      {"cycles", Number(analysis.total_cycles)},
      {"clock_hz", Number(clock)},
      {"total_ms", Milliseconds(analysis.total_cycles, clock)},
      {"busy_ms", Json(Json::Array{Milliseconds(analysis.busy.at(0), clock),
                                   Milliseconds(analysis.busy.at(1), clock),
                                   Milliseconds(analysis.busy.at(2), clock)})},
      {"occupancy_ms", Json(std::move(occupancy))},
      {"blockers_ms", Json(std::move(blockers))},
      {"phases", AggregateJson(ByTag(analysis, options.depth), options)},
      {"categories", AggregateJson(analysis.categories, options)},
      {"largest_gaps", Json(std::move(gaps))}});
}

std::string KernelsText(const Analysis& analysis,
                        const ReportOptions& options) {
  return AggregateTable(ByTag(analysis, options.depth), options,
                        analysis.has_statistics,
                        std::format("tag (depth {})", options.depth));
}

std::string Perfetto(const Analysis& analysis, uint64_t clock_hz) {
  const double us_per_cycle = 1e6 / static_cast<double>(clock_hz);
  std::string text = R"({"displayTimeUnit":"ms","traceEvents":[)";
  bool first = true;
  const auto append = [&](const Json& value) {
    text += first ? "" : ",";
    text += value.Dump();
    first = false;
  };
  for (int e = 0; e < 4; ++e) {
    append(Json(Json::Object{
        {"ph", Json(std::string("M"))},
        {"pid", Number(0)},
        {"tid", Number(static_cast<uint64_t>(e))},
        {"name", Json(std::string("thread_name"))},
        {"args", Json(Json::Object{
                     {"name", Json(std::string(kSourceNames.at(e)))}})}}));
  }
  for (const Event& event : analysis.events) {
    Json::Object args = {{"index", Number(event.index)},
                         {"wait", Number(event.wait)},
                         {"tag", Json(event.tag)}};
    if (event.statistics) {
      args.emplace_back(
          "statistics",
          Json(Json::Array{
              Number(event.statistics->a), Number(event.statistics->b),
              Number(event.statistics->c), Number(event.statistics->d)}));
    }
    append(Json(Json::Object{
        {"name", Json(event.note.empty() ? event.category : event.note)},
        {"cat", Json(event.category)},
        {"ph", Json(std::string("X"))},
        {"pid", Number(0)},
        {"tid", Number(static_cast<uint64_t>(event.source))},
        {"ts", Json(static_cast<double>(event.issue) * us_per_cycle)},
        {"dur",
         Json(static_cast<double>(std::max<uint64_t>(event.Duration(), 1)) *
              us_per_cycle)},
        {"args", Json(std::move(args))}}));
  }
  return text + "]}\n";
}

std::string DiffText(const Analysis& before, const Analysis& after,
                     const ReportOptions& options) {
  const uint64_t clock = options.clock_hz;
  std::string text = std::format(
      "total {:>10.3f} ms -> {:>10.3f} ms ({:+.3f} ms)\n",
      Ms(before.total_cycles, clock), Ms(after.total_cycles, clock),
      Ms(after.total_cycles, clock) - Ms(before.total_cycles, clock));
  const auto table = [&](const std::map<std::string, Aggregate>& a,
                         const std::map<std::string, Aggregate>& b,
                         std::string_view label) {
    std::string rows =
        std::format("\n{:<28s} {:>8s} {:>8s} {:>10s} {:>10s} {:>10s}\n", label,
                    "count", "count'", "span ms", "span' ms", "delta ms");
    std::map<std::string, std::pair<Aggregate, Aggregate>> merged;
    for (const auto& [name, aggregate] : a) {
      merged[name].first = aggregate;
    }
    for (const auto& [name, aggregate] : b) {
      merged[name].second = aggregate;
    }
    std::vector<std::pair<std::string, std::pair<Aggregate, Aggregate>>> sorted(
        merged.begin(), merged.end());
    std::ranges::stable_sort(sorted, [](const auto& x, const auto& y) {
      const auto dx = static_cast<double>(x.second.second.span) -
                      static_cast<double>(x.second.first.span);
      const auto dy = static_cast<double>(y.second.second.span) -
                      static_cast<double>(y.second.first.span);
      return std::fabs(dx) > std::fabs(dy);
    });
    for (const auto& [name, pair] : sorted) {
      rows +=
          std::format("{:<28s} {:>8d} {:>8d} {:>10.3f} {:>10.3f} {:>+10.3f}\n",
                      name.empty() ? "(untagged)" : name.substr(0, 28),
                      pair.first.count, pair.second.count,
                      Ms(pair.first.span, clock), Ms(pair.second.span, clock),
                      Ms(pair.second.span, clock) - Ms(pair.first.span, clock));
    }
    return rows;
  };
  text += table(before.categories, after.categories, "category");
  text += table(ByTag(before, options.depth), ByTag(after, options.depth),
                std::format("tag (depth {})", options.depth));
  return text;
}

namespace {

struct Mechanisms {
  std::vector<double> gemm_fixed;
  LeastSquares reads{3};
  LeastSquares writes{2};
  LeastSquares gather{2};
  std::vector<double> issue_tail;
  LeastSquares drains{2};
  LeastSquares remainder{5};
  std::map<std::string, std::array<double, 2>> residual;
};

double Scaled(uint64_t estimate, uint64_t estimated_beats, uint64_t beats) {
  if (estimated_beats == 0) {
    return static_cast<double>(estimate);
  }
  return static_cast<double>(estimate) * static_cast<double>(beats) /
         static_cast<double>(estimated_beats);
}

void CollectGemm(const Event& event, const iss::Statistics& st, Mechanisms& m) {
  m.gemm_fixed.push_back(static_cast<double>(event.Duration()) -
                         static_cast<double>(st.a));
}

void CollectDmaRead(const Event& event, const iss::Statistics& st,
                    const iss::DmaCommand& command, Mechanisms& m) {
  if (event.category == "GACC") {
    return;
  }
  const iss::DmaActivity estimate = iss::EstimateDma(command);
  m.reads.Add(
      {1.0, static_cast<double>(st.a),
       Scaled(estimate.read_bursts, estimate.read_beats, st.a)},
      static_cast<double>(event.Duration()) - static_cast<double>(st.c));
}

void CollectDmaRest(const Event& event, const iss::Statistics& st,
                    const iss::DmaCommand& command, Mechanisms& m,
                    const iss::TimingParameters& p) {
  const iss::DmaActivity estimate = iss::EstimateDma(command);
  const auto span = static_cast<double>(event.Duration());
  const double fixed = p.dma_fixed_cycles;
  const double read_beat = static_cast<double>(p.dma_read_cycles_per_kilobeat) /
                           iss::kBeatsPerKilobeat;
  const double denied = st.c;
  if (event.category == "GACC") {
    const double entries = 16.0 * static_cast<double>(command.rows);
    const double row_beats = (2.0 * command.row_words) + 1.0;
    const auto other = static_cast<double>(estimate.read_beats);
    const double rows = std::max(0.0, (st.a - other) / row_beats);
    m.gather.Add({entries, rows},
                 span - fixed - denied - (other * read_beat) -
                     (static_cast<double>(estimate.read_bursts) *
                      p.dma_read_burst_cycles) -
                     (static_cast<double>(estimate.gather_fetches) *
                      p.dma_gather_fetch_cycles));
    return;
  }
  if (command.store && st.b > 0) {
    m.writes.Add({static_cast<double>(st.b),
                  Scaled(estimate.write_bursts, estimate.write_beats, st.b)},
                 span - fixed - denied - (st.a * read_beat));
  }
}

void CollectVec(const Event& event, const iss::Statistics& st,
                const iss::VecCommand& command, Mechanisms& m) {
  const auto span = static_cast<double>(event.Duration());
  if (!iss::RunsLibraryProgram(command)) {
    return;
  }
  const iss::VecActivity activity = iss::LibraryActivity(command);
  if (activity.passes == 0) {
    return;
  }
  m.issue_tail.push_back(
      (4.0 *
           static_cast<double>(activity.words + activity.overlap_stall_words) -
       st.a) /
      activity.passes);
  m.drains.Add({static_cast<double>(activity.drains),
                static_cast<double>(activity.drains_quantized)},
               st.b);
  m.remainder.Add({1.0, static_cast<double>(activity.passes),
                   static_cast<double>(activity.salu),
                   static_cast<double>(activity.branches),
                   static_cast<double>(activity.smem)},
                  span - st.a - st.b - st.c);
}

std::string FitGemm(const Mechanisms& m, iss::TimingParameters& p) {
  if (m.gemm_fixed.empty()) {
    return {};
  }
  p.gemm_fixed_cycles = Round(Median(m.gemm_fixed));
  return std::format("gemm_fixed_cycles = {} ({} commands)\n",
                     p.gemm_fixed_cycles, m.gemm_fixed.size());
}

std::string FitReads(const Mechanisms& m, iss::TimingParameters& p) {
  if (m.reads.rows < 3) {
    return {};
  }
  const std::vector<double> fit = m.reads.Solve();
  p.dma_fixed_cycles = Round(fit[0]);
  p.dma_read_cycles_per_kilobeat = Round(fit[1] * iss::kBeatsPerKilobeat);
  p.dma_read_burst_cycles = Round(fit[2]);
  return std::format(
      "dma_fixed_cycles = {}, dma_read_cycles_per_kilobeat = {}, "
      "dma_read_burst_cycles = {} ({} loads)\n",
      p.dma_fixed_cycles, p.dma_read_cycles_per_kilobeat,
      p.dma_read_burst_cycles, m.reads.rows);
}

std::string FitWrites(const Mechanisms& m, iss::TimingParameters& p) {
  std::string report;
  if (m.writes.rows >= 2) {
    const std::vector<double> fit = m.writes.Solve();
    p.dma_write_cycles_per_kilobeat = Round(fit[0] * iss::kBeatsPerKilobeat);
    p.dma_write_burst_cycles = Round(fit[1]);
    report += std::format(
        "dma_write_cycles_per_kilobeat = {}, dma_write_burst_cycles = {} ({} "
        "stores)\n",
        p.dma_write_cycles_per_kilobeat, p.dma_write_burst_cycles,
        m.writes.rows);
  }
  if (m.gather.rows >= 2) {
    const std::vector<double> fit = m.gather.Solve();
    p.dma_gather_entry_cycles = Round(fit[0]);
    p.dma_gather_row_cycles = Round(fit[1]);
    report += std::format(
        "dma_gather_entry_cycles = {}, dma_gather_row_cycles = {} ({} "
        "commands)\n",
        p.dma_gather_entry_cycles, p.dma_gather_row_cycles, m.gather.rows);
  }
  return report;
}

std::string FitVec(const Mechanisms& m, iss::TimingParameters& p) {
  std::string report;
  if (!m.issue_tail.empty()) {
    p.vec_issue_tail_cycles = Round(Median(m.issue_tail));
    report += std::format(
        "vec_issue_tail_cycles = {} (median over {} commands; issue = 4 x "
        "words - tail x passes)\n",
        p.vec_issue_tail_cycles, m.issue_tail.size());
  }
  if (m.drains.rows >= 2) {
    const std::vector<double> fit = m.drains.Solve();
    p.vec_drain_cycles = Round(fit[0]);
    p.vec_drain_quantized_cycles = Round(fit[1]);
    report += std::format(
        "vec_drain_cycles = {}, vec_drain_quantized_cycles = {} ({} "
        "commands)\n",
        p.vec_drain_cycles, p.vec_drain_quantized_cycles, m.drains.rows);
  }
  if (m.remainder.rows >= 5) {
    const std::vector<double> fit = m.remainder.Solve();
    p.vec_fixed_cycles = Round(fit[0]);
    p.vec_pass_cycles = Round(fit[1]);
    p.vec_salu_cycles = Round(fit[2]);
    p.vec_branch_cycles = Round(fit[3]);
    p.vec_smem_cycles = Round(fit[4]);
    report += std::format(
        "vec_fixed_cycles = {}, vec_pass_cycles = {}, vec_salu_cycles = {}, "
        "vec_branch_cycles = {}, vec_smem_cycles = {} (cycles outside issue, "
        "drain and scalar wait over {} fixed-operation commands)\n",
        p.vec_fixed_cycles, p.vec_pass_cycles, p.vec_salu_cycles,
        p.vec_branch_cycles, p.vec_smem_cycles, m.remainder.rows);
  }
  return report;
}

std::string Residuals(const Analysis& analysis, const Stream& stream,
                      const iss::TimingParameters& p) {
  const iss::CostModel model(p);
  std::map<std::string, std::array<double, 2>> totals;
  for (const Event& event : analysis.events) {
    if (!event.statistics) {
      continue;
    }
    const iss::Statistics& st = *event.statistics;
    double predicted = 0.0;
    if (event.opcode == kOpGemm) {
      predicted = static_cast<double>(
          model.Gemm(iss::DecodeGemm(stream.instructions.at(event.index).word))
              .cycles);
    } else if (event.opcode == kOpDmaLoad || event.opcode == kOpDmaStore) {
      const iss::DmaCommand command =
          iss::DecodeDma(stream.instructions.at(event.index).word);
      iss::DmaActivity activity = iss::EstimateDma(command);
      activity.write_bursts = static_cast<uint64_t>(
          Scaled(activity.write_bursts, activity.write_beats, st.b));
      if (activity.gather_entries > 0) {
        const uint64_t row_beats = (2 * uint64_t{command.row_words}) + 1;
        const uint64_t rows = st.a > activity.read_beats
                                  ? (st.a - activity.read_beats) / row_beats
                                  : 0;
        activity.gather_rows = rows;
        activity.gather_row_beats = rows * row_beats;
      } else {
        activity.read_bursts = static_cast<uint64_t>(
            Scaled(activity.read_bursts, activity.read_beats, st.a));
        activity.read_beats = st.a;
      }
      activity.write_beats = st.b;
      predicted = static_cast<double>(model.Dma(activity).cycles + st.c);
    } else if (event.opcode == kOpVec) {
      const iss::VecCommand command =
          iss::DecodeVec(stream.instructions.at(event.index).word);
      if (!iss::RunsLibraryProgram(command)) {
        predicted = p.vec_fixed_cycles + st.a + st.b + st.c +
                    (static_cast<double>(st.d) * p.vec_pass_cycles);
      } else {
        predicted = static_cast<double>(
            model.Vec(iss::LibraryActivity(command)).cycles);
      }
    } else {
      continue;
    }
    totals[event.category][0] += static_cast<double>(event.Duration());
    totals[event.category][1] += predicted;
  }
  std::string text = std::format("\n{:<28s} {:>12s} {:>12s} {:>8s}\n",
                                 "category", "measured", "model", "error");
  for (const auto& [category, pair] : totals) {
    text += std::format(
        "{:<28s} {:>12.0f} {:>12.0f} {:>+7.1f}%\n", category, pair[0], pair[1],
        pair[0] > 0.0 ? 100.0 * (pair[1] - pair[0]) / pair[0] : 0.0);
  }
  return text;
}

}  // namespace

Calibration Calibrate(const std::vector<CalibrationInput>& inputs) {
  iss::TimingParameters p;
  std::string report;
  Mechanisms m;
  uint64_t first_issue = UINT64_MAX;
  bool statistics = false;
  for (const CalibrationInput& input : inputs) {
    first_issue = std::min(first_issue, input.analysis->first_issue);
    statistics = statistics || input.analysis->has_statistics;
    for (const Event& event : input.analysis->events) {
      if (!event.statistics) {
        continue;
      }
      if (event.opcode == kOpGemm) {
        CollectGemm(event, *event.statistics, m);
      } else if (event.opcode == kOpDmaLoad) {
        CollectDmaRead(
            event, *event.statistics,
            iss::DecodeDma(input.stream->instructions.at(event.index).word), m);
      }
    }
  }
  p.start_cycles = static_cast<uint32_t>(std::min<uint64_t>(
      first_issue == UINT64_MAX ? 0 : first_issue, UINT32_MAX));
  if (!statistics) {
    report =
        "the trace has no statistics records; calibration needs a trace "
        "captured with TRACE_CTRL bit 2\n";
    return {.parameters = p, .report = report};
  }
  report += FitGemm(m, p);
  report += FitReads(m, p);
  for (const CalibrationInput& input : inputs) {
    for (const Event& event : input.analysis->events) {
      if (!event.statistics) {
        continue;
      }
      const iss::Word128& word =
          input.stream->instructions.at(event.index).word;
      if (event.opcode == kOpDmaLoad || event.opcode == kOpDmaStore) {
        CollectDmaRest(event, *event.statistics, iss::DecodeDma(word), m, p);
      } else if (event.opcode == kOpVec) {
        CollectVec(event, *event.statistics, iss::DecodeVec(word), m);
      }
    }
  }
  report += FitWrites(m, p);
  report += FitVec(m, p);
  report += std::format("start_cycles = {}\n", p.start_cycles);
  for (const CalibrationInput& input : inputs) {
    report += std::format("\nstream {}", input.stream->name);
    report += Residuals(*input.analysis, *input.stream, p);
  }
  return {.parameters = p, .report = report};
}

}  // namespace hotarubi::trace
