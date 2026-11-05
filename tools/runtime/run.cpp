#include <charconv>
#include <chrono>
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
#include "hotarubi/runtime.hpp"
#include "hotarubi/safetensors.hpp"
#include "smolvla_ps.hpp"
#include "sparsedrive_ps.hpp"

namespace hotarubi {
namespace {

using Options = std::map<std::string, std::string, std::less<>>;

constexpr std::string_view kUsage =
    "usage: hotarubi_run smolvla --program DIR --language FILE --input DIR "
    "--output DIR [--timeout MS] [--seed N]\n"
    "       hotarubi_run sparsedrive --program DIR --input DIR --output DIR "
    "[--timeout MS]";
constexpr std::string_view kUioClass = "/sys/class/uio";
constexpr std::string_view kUioName = "npu";
constexpr std::string_view kBufferClass = "/sys/class/u-dma-buf/udmabuf0";
constexpr std::string_view kBufferDevice = "/dev/udmabuf0";
constexpr uint64_t kDefaultTimeoutMs = 60000;

struct Settings {
  std::filesystem::path program;
  std::filesystem::path input;
  std::filesystem::path output;
  std::chrono::milliseconds timeout{0};
};

std::expected<Options, std::string> ParseOptions(
    std::span<const char* const> args) {
  static const std::set<std::string, std::less<>> kKnown = {
      "program", "language", "input", "output", "timeout", "seed"};
  Options options;
  for (size_t i = 0; i < args.size(); i += 2) {
    const std::string_view flag = args[i];
    if (!flag.starts_with("--") || !kKnown.contains(flag.substr(2)) ||
        i + 1 >= args.size()) {
      return std::unexpected(std::format("bad argument {}", flag));
    }
    options[std::string(flag.substr(2))] = args[i + 1];
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
  return Settings{.program = *program,
                  .input = *input,
                  .output = *output,
                  .timeout = std::chrono::milliseconds(*timeout)};
}

std::expected<std::vector<std::filesystem::path>, std::string> Prepare(
    Runtime& runtime, const Settings& settings) {
  auto program = LoadProgram(settings.program);
  if (!program) {
    return std::unexpected(program.error());
  }
  if (auto loaded = runtime.Load(std::move(*program)); !loaded) {
    return std::unexpected(loaded.error());
  }
  std::error_code error;
  std::filesystem::create_directories(settings.output, error);
  if (error) {
    return std::unexpected(std::format(
        "cannot create {}: {}", settings.output.string(), error.message()));
  }
  return FramePaths(settings.input);
}

std::string Elapsed(std::chrono::steady_clock::time_point start) {
  return std::format("{} ms",
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count());
}

std::expected<void, std::string> RunSmolVlaFrame(
    Runtime& runtime, const Settings& settings,
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
  const auto result =
      smolvla::RunFrame(runtime, *frame, *language, settings.timeout);
  if (!result) {
    return std::unexpected(result.error());
  }
  std::println("{}: {} cycles, {}", path.filename().string(),
               result->counters.cycles, Elapsed(start));
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
  const auto frames = Prepare(runtime, *settings);
  if (!frames) {
    return std::unexpected(frames.error());
  }
  std::mt19937_64 generator(*seed);
  for (const std::filesystem::path& path : *frames) {
    if (auto done =
            RunSmolVlaFrame(runtime, *settings, *table, generator, path);
        !done) {
      return std::unexpected(
          std::format("{}: {}", path.string(), done.error()));
    }
  }
  return {};
}

std::expected<void, std::string> RunSparseDriveFrame(
    Runtime& runtime, const Settings& settings,
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
  const auto result =
      sparsedrive::RunFrame(runtime, stream, *frame, settings.timeout);
  if (!result) {
    return std::unexpected(result.error());
  }
  std::println("{}: {}, {} cycles, {}", path.filename().string(), stream,
               result->counters.cycles, Elapsed(start));
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
  const auto frames = Prepare(runtime, *settings);
  if (!frames) {
    return std::unexpected(frames.error());
  }
  sparsedrive::SceneTracker tracker;
  for (const std::filesystem::path& path : *frames) {
    if (auto done = RunSparseDriveFrame(runtime, *settings, tracker, path);
        !done) {
      return std::unexpected(
          std::format("{}: {}", path.string(), done.error()));
    }
  }
  return {};
}

std::expected<void, std::string> Run(std::string_view model,
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
  return model == "smolvla" ? RunSmolVla(runtime, options)
                            : RunSparseDrive(runtime, options);
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
