#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <expected>
#include <filesystem>
#include <format>
#include <map>
#include <optional>
#include <print>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "analysis.hpp"
#include "input.hpp"
#include "predict.hpp"
#include "report.hpp"
#include "trace_record.hpp"

namespace hotarubi::trace {
namespace {

using Options = std::map<std::string, std::string, std::less<>>;

constexpr std::string_view kUsage =
    "usage: hotarubi_trace check    --program DIR --stream NAME --trace FILE "
    "[--counters FILE]\n"
    "       hotarubi_trace report   --program DIR --stream NAME --trace FILE "
    "[--counters FILE] [--depth N] [--clock-hz N] [--gaps N] [--json FILE]\n"
    "       hotarubi_trace kernels  --program DIR --stream NAME --trace FILE "
    "[--counters FILE] [--depth N] [--clock-hz N]\n"
    "       hotarubi_trace perfetto --program DIR --stream NAME --trace FILE "
    "[--counters FILE] [--clock-hz N] --out FILE\n"
    "       hotarubi_trace diff     --program DIR --stream NAME --trace FILE "
    "--trace2 FILE [--program2 DIR] [--depth N] [--clock-hz N]\n"
    "       hotarubi_trace predict  --program DIR --stream NAME --out DIR "
    "[--stats] [--accesses] [--timing NAME=VALUE,...]\n"
    "       hotarubi_trace calibrate --program DIR --stream NAME --trace FILE "
    "[--counters FILE] [--program2 DIR --stream2 NAME --trace2 FILE]";

std::expected<Options, std::string> ParseOptions(
    std::span<const char* const> args) {
  static const std::set<std::string, std::less<>> kValued = {
      "program", "program2", "stream", "stream2", "trace", "trace2", "counters",
      "depth",   "clock-hz", "gaps",   "json",    "out",   "timing"};
  static const std::set<std::string, std::less<>> kSwitches = {"stats",
                                                               "accesses"};
  Options options;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string_view flag = args[i];
    if (!flag.starts_with("--")) {
      return std::unexpected(std::format("bad argument {}", flag));
    }
    const std::string_view name = flag.substr(2);
    if (kSwitches.contains(name)) {
      options[std::string(name)] = "1";
    } else if (kValued.contains(name) && i + 1 < args.size()) {
      options[std::string(name)] = args[++i];
    } else {
      return std::unexpected(std::format("bad argument {}", flag));
    }
  }
  return options;
}

std::expected<std::string, std::string> Text(const Options& options,
                                             std::string_view name) {
  const auto found = options.find(name);
  if (found == options.end()) {
    return std::unexpected(std::format("--{} is required", name));
  }
  return found->second;
}

std::expected<uint64_t, std::string> Number(const Options& options,
                                            std::string_view name,
                                            uint64_t fallback) {
  const auto found = options.find(name);
  if (found == options.end()) {
    return fallback;
  }
  const std::string& digits = found->second;
  uint64_t value = 0;
  const char* const end = digits.data() + digits.size();
  const auto [last, error] = std::from_chars(digits.data(), end, value);
  if (digits.empty() || error != std::errc{} || last != end) {
    return std::unexpected(std::format("--{} needs a number", name));
  }
  return value;
}

std::expected<ReportOptions, std::string> ReadReportOptions(
    const Options& options) {
  const auto depth = Number(options, "depth", 1);
  const auto clock = Number(options, "clock-hz", kDefaultClockHz);
  const auto gaps = Number(options, "gaps", 20);
  for (const auto* value : {&depth, &clock, &gaps}) {
    if (!*value) {
      return std::unexpected(value->error());
    }
  }
  if (*clock == 0) {
    return std::unexpected("--clock-hz must be positive");
  }
  return ReportOptions{.depth = static_cast<size_t>(*depth),
                       .clock_hz = *clock,
                       .gaps = static_cast<size_t>(*gaps)};
}

struct Inputs {
  Stream stream;
  std::vector<iss::TraceRecord> records;
  std::optional<CountersFile> counters;
};

std::expected<Inputs, std::string> LoadInputs(const Options& options,
                                              std::string_view program_key,
                                              std::string_view trace_key) {
  const auto program = Text(options, program_key);
  const auto stream_name = Text(
      options, trace_key == "trace2" && options.contains("stream2") ? "stream2"
                                                                    : "stream");
  const auto trace = Text(options, trace_key);
  for (const auto* text : {&program, &stream_name, &trace}) {
    if (!*text) {
      return std::unexpected(text->error());
    }
  }
  const auto loaded = LoadCompiled(*program);
  if (!loaded) {
    return std::unexpected(loaded.error());
  }
  auto stream = LoadStream(*loaded, *stream_name);
  if (!stream) {
    return std::unexpected(stream.error());
  }
  auto records = LoadRecords(*trace);
  if (!records) {
    return std::unexpected(records.error());
  }
  Inputs inputs{.stream = std::move(*stream),
                .records = std::move(*records),
                .counters = std::nullopt};
  if (const auto counters = options.find("counters");
      counters != options.end() && trace_key == "trace") {
    auto file = LoadCounters(counters->second);
    if (!file) {
      return std::unexpected(file.error());
    }
    inputs.counters = std::move(*file);
  }
  return inputs;
}

std::expected<Analysis, std::string> LoadAnalysis(const Options& options,
                                                  std::string_view program_key,
                                                  std::string_view trace_key) {
  const auto inputs = LoadInputs(options, program_key, trace_key);
  if (!inputs) {
    return std::unexpected(inputs.error());
  }
  return Analyze(inputs->stream, inputs->records, inputs->counters);
}

std::expected<void, std::string> RunCheck(const Options& options) {
  const auto inputs = LoadInputs(options, "program", "trace");
  if (!inputs) {
    return std::unexpected(inputs.error());
  }
  if (auto checked = Check(inputs->stream, inputs->records, inputs->counters);
      !checked) {
    return checked;
  }
  std::println("{} records for {} instructions of stream {}: consistent",
               inputs->records.size(), inputs->stream.count,
               inputs->stream.name);
  return {};
}

std::expected<void, std::string> RunReport(const Options& options,
                                           bool kernels) {
  const auto report_options = ReadReportOptions(options);
  if (!report_options) {
    return std::unexpected(report_options.error());
  }
  const auto analysis = LoadAnalysis(options, "program", "trace");
  if (!analysis) {
    return std::unexpected(analysis.error());
  }
  if (kernels) {
    std::print("{}", KernelsText(*analysis, *report_options));
    return {};
  }
  std::print("{}", ReportText(*analysis, *report_options));
  if (const auto json = options.find("json"); json != options.end()) {
    return WriteText(json->second,
                     ReportJson(*analysis, *report_options).Dump() + "\n");
  }
  return {};
}

std::expected<void, std::string> RunPerfetto(const Options& options) {
  const auto report_options = ReadReportOptions(options);
  const auto out = Text(options, "out");
  if (!report_options) {
    return std::unexpected(report_options.error());
  }
  if (!out) {
    return std::unexpected(out.error());
  }
  const auto analysis = LoadAnalysis(options, "program", "trace");
  if (!analysis) {
    return std::unexpected(analysis.error());
  }
  return WriteText(*out, Perfetto(*analysis, report_options->clock_hz));
}

std::expected<void, std::string> RunDiff(const Options& options) {
  const auto report_options = ReadReportOptions(options);
  if (!report_options) {
    return std::unexpected(report_options.error());
  }
  const auto before = LoadAnalysis(options, "program", "trace");
  if (!before) {
    return std::unexpected(before.error());
  }
  const auto after = LoadAnalysis(
      options, options.contains("program2") ? "program2" : "program", "trace2");
  if (!after) {
    return std::unexpected(after.error());
  }
  std::print("{}", DiffText(*before, *after, *report_options));
  return {};
}

std::expected<void, std::string> RunPredict(const Options& options) {
  const auto program = Text(options, "program");
  const auto stream = Text(options, "stream");
  const auto out = Text(options, "out");
  for (const auto* text : {&program, &stream, &out}) {
    if (!*text) {
      return std::unexpected(text->error());
    }
  }
  const auto loaded = LoadCompiled(*program);
  if (!loaded) {
    return std::unexpected(loaded.error());
  }
  const bool statistics = options.contains("stats");
  const auto timing = options.contains("timing")
                          ? iss::ParseTiming(options.at("timing"))
                          : iss::TimingParameters{};
  if (!timing) {
    return std::unexpected(timing.error());
  }
  const bool accesses = options.contains("accesses");
  const auto prediction =
      Predict(*loaded, *stream, statistics, *timing, accesses);
  if (!prediction) {
    return std::unexpected(prediction.error());
  }
  std::error_code error;
  std::filesystem::create_directories(*out, error);
  if (error) {
    return std::unexpected(
        std::format("cannot create {}: {}", *out, error.message()));
  }
  const std::filesystem::path directory(*out);
  const std::string_view bytes(
      reinterpret_cast<const char*>(prediction->records.data()),
      prediction->records.size());
  if (auto written = WriteText(directory / (*stream + ".trace.bin"), bytes);
      !written) {
    return written;
  }
  if (auto written = WriteText(
          directory / (*stream + ".counters.json"),
          CountersJson(*stream, statistics, prediction->counters, "predicted"));
      !written) {
    return written;
  }
  if (accesses) {
    if (auto written = WriteText(directory / (*stream + ".accesses.jsonl"),
                                 prediction->accesses);
        !written) {
      return written;
    }
  }
  std::println("{}: {} records, {} cycles", *stream,
               prediction->counters.trace_count, prediction->counters.cycles);
  return {};
}

std::expected<void, std::string> RunCalibrate(const Options& options) {
  std::vector<Inputs> inputs;
  std::vector<Analysis> analyses;
  auto first = LoadInputs(options, "program", "trace");
  if (!first) {
    return std::unexpected(first.error());
  }
  inputs.push_back(std::move(*first));
  if (options.contains("trace2")) {
    auto second = LoadInputs(options, "program2", "trace2");
    if (!second) {
      return std::unexpected(second.error());
    }
    inputs.push_back(std::move(*second));
  }
  for (const Inputs& input : inputs) {
    auto analysis = Analyze(input.stream, input.records, input.counters);
    if (!analysis) {
      return std::unexpected(analysis.error());
    }
    analyses.push_back(std::move(*analysis));
  }
  std::vector<CalibrationInput> calibration;
  calibration.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    calibration.push_back(
        {.analysis = &analyses[i], .stream = &inputs[i].stream});
  }
  std::print("{}", Calibrate(calibration).report);
  return {};
}

int Main(std::span<const char* const> args) {
  if (args.size() < 2) {
    std::println(stderr, "{}", kUsage);
    return 2;
  }
  const std::string_view command = args[1];
  const auto options = ParseOptions(args.subspan(2));
  if (!options) {
    std::println(stderr, "{}\n{}", options.error(), kUsage);
    return 2;
  }
  std::expected<void, std::string> result;
  if (command == "check") {
    result = RunCheck(*options);
  } else if (command == "report") {
    result = RunReport(*options, false);
  } else if (command == "kernels") {
    result = RunReport(*options, true);
  } else if (command == "perfetto") {
    result = RunPerfetto(*options);
  } else if (command == "diff") {
    result = RunDiff(*options);
  } else if (command == "predict") {
    result = RunPredict(*options);
  } else if (command == "calibrate") {
    result = RunCalibrate(*options);
  } else {
    std::println(stderr, "unknown command {}\n{}", command, kUsage);
    return 2;
  }
  if (!result) {
    std::println(stderr, "{}", result.error());
    return 1;
  }
  return 0;
}

}  // namespace
}  // namespace hotarubi::trace

int main(int argc, char** argv) {
  try {
    return hotarubi::trace::Main(
        std::span<const char* const>(argv, static_cast<size_t>(argc)));
  } catch (const std::exception& error) {
    std::fputs(error.what(), stderr);
    std::fputs("\n", stderr);
    return 1;
  }
}
