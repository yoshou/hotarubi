#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <optional>
#include <print>
#include <random>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "frame_files.hpp"
#include "hotarubi/driver.hpp"
#include "hotarubi/json.hpp"
#include "hotarubi/runtime.hpp"
#include "hotarubi/safetensors.hpp"
#include "hotarubi/sha256.hpp"
#include "iss_bus.hpp"
#include "machine.hpp"
#include "smolvla_ps.hpp"
#include "sparsedrive_ps.hpp"

namespace hotarubi {
namespace {

using Options = std::map<std::string, std::string, std::less<>>;

constexpr std::string_view kUsage =
    "usage: hotarubi_run smolvla --program DIR --language FILE --input DIR "
    "--output DIR [--timeout MS] [--seed N] [--trace DIR] [--trace-stats] "
    "[--device board|iss]\n"
    "       hotarubi_run sparsedrive --program DIR --input DIR --output DIR "
    "[--timeout MS] [--trace DIR] [--trace-stats] [--device board|iss]";
constexpr std::string_view kUioClass = "/sys/class/uio";
constexpr std::string_view kUioName = "npu";
constexpr std::string_view kBufferClass = "/sys/class/u-dma-buf/udmabuf0";
constexpr std::string_view kBufferDevice = "/dev/udmabuf0";
constexpr std::string_view kBitstream = "/lib/firmware/xilinx/npu/npu.bit.bin";
constexpr std::string_view kClockRate = "/sys/devices/platform/fclk0/set_rate";
constexpr uint64_t kDefaultTimeoutMs = 60000;

struct Settings {
  std::filesystem::path program;
  std::filesystem::path input;
  std::filesystem::path output;
  std::chrono::milliseconds timeout{0};
  std::optional<std::filesystem::path> trace;
  bool trace_statistics = false;
};

struct Provenance {
  std::string image;
  std::optional<std::string> bitstream;
  std::optional<uint64_t> clock_hz;
};

std::expected<Options, std::string> ParseOptions(
    std::span<const char* const> args) {
  static const std::set<std::string, std::less<>> kKnown = {
      "program", "language", "input", "output",
      "timeout", "seed",     "trace", "device"};
  static const std::set<std::string, std::less<>> kSwitches = {"trace-stats"};
  Options options;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string_view flag = args[i];
    if (!flag.starts_with("--")) {
      return std::unexpected(std::format("bad argument {}", flag));
    }
    const std::string_view name = flag.substr(2);
    if (kSwitches.contains(name)) {
      options[std::string(name)] = "1";
    } else if (kKnown.contains(name) && i + 1 < args.size()) {
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

std::expected<Settings, std::string> ReadSettings(const Options& options) {
  const auto program = Text(options, "program");
  const auto input = Text(options, "input");
  const auto output = Text(options, "output");
  for (const auto* text : {&program, &input, &output}) {
    if (!*text) {
      return std::unexpected(text->error());
    }
  }
  const auto timeout = Number(options, "timeout", kDefaultTimeoutMs);
  if (!timeout) {
    return std::unexpected(timeout.error());
  }
  Settings settings{.program = *program,
                    .input = *input,
                    .output = *output,
                    .timeout = std::chrono::milliseconds(*timeout),
                    .trace = std::nullopt,
                    .trace_statistics = options.contains("trace-stats")};
  if (const auto trace = options.find("trace"); trace != options.end()) {
    settings.trace = trace->second;
  } else if (settings.trace_statistics) {
    return std::unexpected("--trace-stats needs --trace");
  }
  return settings;
}

std::expected<void, std::string> MakeDirectory(
    const std::filesystem::path& path) {
  std::error_code error;
  std::filesystem::create_directories(path, error);
  if (error) {
    return std::unexpected(
        std::format("cannot create {}: {}", path.string(), error.message()));
  }
  return {};
}

std::optional<uint64_t> ReadClockRate() {
  std::ifstream file{std::filesystem::path(kClockRate)};
  uint64_t rate = 0;
  if (file >> rate) {
    return rate;
  }
  return std::nullopt;
}

struct Prepared {
  std::vector<std::filesystem::path> frames;
  Provenance provenance;
};

std::expected<Prepared, std::string> Prepare(Runtime& runtime,
                                             const Settings& settings) {
  auto program = LoadProgram(settings.program);
  if (!program) {
    return std::unexpected(program.error());
  }
  Provenance provenance{.image = Sha256Hex(program->image),
                        .bitstream = std::nullopt,
                        .clock_hz = ReadClockRate()};
  if (const auto bitstream = Sha256File(kBitstream); bitstream) {
    provenance.bitstream = *bitstream;
  }
  if (auto loaded = runtime.Load(std::move(*program)); !loaded) {
    return std::unexpected(loaded.error());
  }
  if (auto made = MakeDirectory(settings.output); !made) {
    return std::unexpected(made.error());
  }
  if (settings.trace) {
    if (auto made = MakeDirectory(*settings.trace); !made) {
      return std::unexpected(made.error());
    }
  }
  auto frames = FramePaths(settings.input);
  if (!frames) {
    return std::unexpected(frames.error());
  }
  return Prepared{.frames = std::move(*frames),
                  .provenance = std::move(provenance)};
}

std::string Elapsed(std::chrono::steady_clock::time_point start) {
  return std::format("{} ms",
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count());
}

TraceRequest TraceOf(const Settings& settings) {
  return {.enabled = settings.trace.has_value(),
          .statistics = settings.trace_statistics};
}

struct FrameRecord {
  std::string stream;
  Counters counters;
  Timing timing;
  std::chrono::nanoseconds wall{0};
};

Json Number(uint64_t value) { return Json(static_cast<double>(value)); }

Json Nanoseconds(std::chrono::nanoseconds value) {
  return Number(static_cast<uint64_t>(value.count()));
}

std::expected<void, std::string> WriteBytes(const std::filesystem::path& path,
                                            std::span<const std::byte> bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!file) {
    return std::unexpected(std::format("cannot write {}", path.string()));
  }
  return {};
}

std::expected<void, std::string> SaveTrace(Runtime& runtime,
                                           const Settings& settings,
                                           const Provenance& provenance,
                                           const std::filesystem::path& frame,
                                           const FrameRecord& record) {
  if (!settings.trace) {
    return {};
  }
  const std::string stem = frame.stem().string();
  const auto records = runtime.ReadTrace(record.counters.trace_count);
  if (!records) {
    return std::unexpected(records.error());
  }
  if (auto written =
          WriteBytes(*settings.trace / (stem + ".trace.bin"), *records);
      !written) {
    return written;
  }
  const auto input = Sha256File(frame);
  if (!input) {
    return std::unexpected(input.error());
  }
  const Counters& c = record.counters;
  Json::Object counters = {{"issued", Number(c.issued)},
                           {"cycles", Number(c.cycles)},
                           {"gemm_cycles", Number(c.gemm_cycles)},
                           {"vec_cycles", Number(c.vec_cycles)},
                           {"dma_cycles", Number(c.dma_cycles)},
                           {"stall_cycles", Number(c.stall_cycles)},
                           {"starve_cycles", Number(c.starve_cycles)},
                           {"trace_count", Number(c.trace_count)}};
  Json::Object timing = {{"inputs_ns", Nanoseconds(record.timing.inputs)},
                         {"run_ns", Nanoseconds(record.timing.run)},
                         {"signals_ns", Nanoseconds(record.timing.signals)},
                         {"outputs_ns", Nanoseconds(record.timing.outputs)},
                         {"frame_ns", Nanoseconds(record.wall)}};
  Json::Object hashes = {{"image", Json(provenance.image)},
                         {"input", Json(*input)}};
  if (provenance.bitstream) {
    hashes.emplace_back("bitstream", Json(*provenance.bitstream));
  }
  Json::Object root = {{"stream", Json(record.stream)},
                       {"trace_statistics", Json(settings.trace_statistics)},
                       {"counters", Json(std::move(counters))},
                       {"timing", Json(std::move(timing))},
                       {"sha256", Json(std::move(hashes))}};
  if (provenance.clock_hz) {
    root.emplace_back("pl_clock_hz", Number(*provenance.clock_hz));
  }
  const std::string text = Json(std::move(root)).Dump() + "\n";
  return WriteBytes(*settings.trace / (stem + ".counters.json"),
                    std::as_bytes(std::span<const char>(text)));
}

std::expected<void, std::string> RunSmolVlaFrame(
    Runtime& runtime, const Settings& settings, const Provenance& provenance,
    const smolvla::LanguageTable& table, std::mt19937_64& generator,
    const std::filesystem::path& path) {
  const auto file = SafeTensors::Open(path);
  if (!file) {
    return std::unexpected(file.error());
  }
  const auto frame = smolvla::ReadFrame(*file, generator);
  if (!frame) {
    return std::unexpected(frame.error());
  }
  const auto language = table.Rows(frame->task);
  if (!language) {
    return std::unexpected(language.error());
  }
  const auto start = std::chrono::steady_clock::now();
  const auto result = smolvla::RunFrame(runtime, *frame, *language,
                                        settings.timeout, TraceOf(settings));
  if (!result) {
    return std::unexpected(result.error());
  }
  const auto wall = std::chrono::steady_clock::now() - start;
  std::println("{}: {} cycles, {}", path.filename().string(),
               result->counters.cycles, Elapsed(start));
  if (auto saved = SaveTrace(
          runtime, settings, provenance, path,
          {.stream = "inference",
           .counters = result->counters,
           .timing = result->timing,
           .wall = std::chrono::duration_cast<std::chrono::nanoseconds>(wall)});
      !saved) {
    return saved;
  }
  return WriteSafeTensors(settings.output / path.filename(),
                          smolvla::OutputTensors(*frame, *result),
                          {{"task", frame->task}});
}

std::expected<void, std::string> RunSmolVla(Runtime& runtime,
                                            const Options& options) {
  const auto settings = ReadSettings(options);
  const auto language = Text(options, "language");
  const auto seed = Number(options, "seed", 0);
  if (!settings) {
    return std::unexpected(settings.error());
  }
  if (!language) {
    return std::unexpected(language.error());
  }
  if (!seed) {
    return std::unexpected(seed.error());
  }
  const auto table = smolvla::LanguageTable::Open(*language);
  if (!table) {
    return std::unexpected(table.error());
  }
  const auto prepared = Prepare(runtime, *settings);
  if (!prepared) {
    return std::unexpected(prepared.error());
  }
  std::mt19937_64 generator(*seed);
  for (const std::filesystem::path& path : prepared->frames) {
    if (auto done = RunSmolVlaFrame(runtime, *settings, prepared->provenance,
                                    *table, generator, path);
        !done) {
      return std::unexpected(
          std::format("{}: {}", path.string(), done.error()));
    }
  }
  return {};
}

std::expected<void, std::string> RunSparseDriveFrame(
    Runtime& runtime, const Settings& settings, const Provenance& provenance,
    sparsedrive::SceneTracker& tracker, const std::filesystem::path& path) {
  const auto file = SafeTensors::Open(path);
  if (!file) {
    return std::unexpected(file.error());
  }
  const auto frame = sparsedrive::ReadFrame(*file);
  if (!frame) {
    return std::unexpected(frame.error());
  }
  const std::string stream(tracker.Stream(frame->scene));
  const auto start = std::chrono::steady_clock::now();
  const auto result = sparsedrive::RunFrame(
      runtime, stream, *frame, settings.timeout, TraceOf(settings));
  if (!result) {
    return std::unexpected(result.error());
  }
  const auto wall = std::chrono::steady_clock::now() - start;
  std::println("{}: {}, {} cycles, {}", path.filename().string(), stream,
               result->counters.cycles, Elapsed(start));
  if (auto saved = SaveTrace(
          runtime, settings, provenance, path,
          {.stream = stream,
           .counters = result->counters,
           .timing = result->timing,
           .wall = std::chrono::duration_cast<std::chrono::nanoseconds>(wall)});
      !saved) {
    return saved;
  }
  return WriteSafeTensors(settings.output / path.filename(),
                          sparsedrive::OutputTensors(*frame, *result),
                          {{"token", frame->token}, {"stream", stream}});
}

std::expected<void, std::string> RunSparseDrive(Runtime& runtime,
                                                const Options& options) {
  const auto settings = ReadSettings(options);
  if (!settings) {
    return std::unexpected(settings.error());
  }
  const auto prepared = Prepare(runtime, *settings);
  if (!prepared) {
    return std::unexpected(prepared.error());
  }
  sparsedrive::SceneTracker tracker;
  for (const std::filesystem::path& path : prepared->frames) {
    if (auto done = RunSparseDriveFrame(runtime, *settings,
                                        prepared->provenance, tracker, path);
        !done) {
      return std::unexpected(
          std::format("{}: {}", path.string(), done.error()));
    }
  }
  return {};
}

std::expected<void, std::string> RunModel(Runtime& runtime,
                                          std::string_view model,
                                          const Options& options) {
  return model == "smolvla" ? RunSmolVla(runtime, options)
                            : RunSparseDrive(runtime, options);
}

std::expected<void, std::string> RunOnBoard(std::string_view model,
                                            const Options& options) {
  const auto uio = FindUio(kUioClass, kUioName);
  if (!uio) {
    return std::unexpected(uio.error());
  }
  const auto bus = UioBus::Open(*uio);
  if (!bus) {
    return std::unexpected(bus.error());
  }
  const auto buffer = SharedBuffer::Open(kBufferClass, kBufferDevice);
  if (!buffer) {
    return std::unexpected(buffer.error());
  }
  Device device(**bus);
  Runtime runtime(device, buffer->Bytes(), buffer->PhysicalAddress());
  return RunModel(runtime, model, options);
}

std::expected<void, std::string> RunOnSimulator(std::string_view model,
                                                const Options& options) {
  const auto directory = Text(options, "program");
  if (!directory) {
    return std::unexpected(directory.error());
  }
  const auto program = LoadProgram(*directory);
  if (!program) {
    return std::unexpected(program.error());
  }
  std::vector<std::byte> memory(program->image.size());
  iss::Machine machine(memory, program->base);
  iss::IssBus bus(machine);
  Device device(bus);
  Runtime runtime(device, memory, program->base);
  return RunModel(runtime, model, options);
}

std::expected<void, std::string> Run(std::string_view model,
                                     const Options& options) {
  const auto device = options.find("device");
  if (device == options.end() || device->second == "board") {
    return RunOnBoard(model, options);
  }
  if (device->second == "iss") {
    return RunOnSimulator(model, options);
  }
  return std::unexpected(std::format("unknown device {}", device->second));
}

int Main(std::span<const char* const> args) {
  if (args.size() < 2 || (std::string_view(args[1]) != "smolvla" &&
                          std::string_view(args[1]) != "sparsedrive")) {
    std::println(stderr, "{}", kUsage);
    return 2;
  }
  const auto options = ParseOptions(args.subspan(2));
  if (!options) {
    std::println(stderr, "{}\n{}", options.error(), kUsage);
    return 2;
  }
  if (auto result = Run(args[1], *options); !result) {
    std::println(stderr, "{}", result.error());
    return 1;
  }
  return 0;
}

}  // namespace
}  // namespace hotarubi

int main(int argc, char** argv) {
  try {
    return hotarubi::Main(
        std::span<const char* const>(argv, static_cast<size_t>(argc)));
  } catch (const std::exception& error) {
    std::fputs(error.what(), stderr);
    std::fputs("\n", stderr);
    return 1;
  }
}
