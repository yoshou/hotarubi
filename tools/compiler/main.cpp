#include <charconv>
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
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>

#include "checkpoint.hpp"
#include "hotarubi/safetensors.hpp"
#include "output.hpp"
#include "smolvla.hpp"
#include "sparsedrive.hpp"

namespace hotarubi::compiler {
namespace {

using Options = std::map<std::string, std::string, std::less<>>;

constexpr uint64_t kDefaultWindow = 64;

constexpr std::string_view kUsage =
    "usage: hotarubi_compile smolvla --quant FILE --model FILE --base ADDRESS "
    "--lang-tokens N [--cameras N] [--steps N] [--schedule-window N] "
    "--output DIR [--listing DIR]\n"
    "       hotarubi_compile sparsedrive --quant FILE --checkpoint FILE "
    "--base ADDRESS [--schedule-window N] --output DIR [--listing DIR]";

std::expected<Options, std::string> ParseOptions(
    std::span<const char* const> args) {
  static const std::set<std::string, std::less<>> kKnown = {
      "quant",   "model", "checkpoint", "base",    "lang-tokens",
      "cameras", "steps", "output",     "listing", "schedule-window"};
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
                                            std::optional<uint64_t> fallback) {
  const auto found = options.find(name);
  if (found == options.end()) {
    if (fallback.has_value()) {
      return *fallback;
    }
    return std::unexpected(std::format("--{} is required", name));
  }
  std::string_view text = found->second;
  int base = 10;
  if (text.starts_with("0x") || text.starts_with("0X")) {
    text.remove_prefix(2);
    base = 16;
  }
  const std::string digits(text);
  uint64_t value = 0;
  const char* const end = digits.data() + digits.size();
  const auto [last, error] = std::from_chars(digits.data(), end, value, base);
  if (digits.empty() || error != std::errc{} || last != end) {
    return std::unexpected(std::format("--{} needs a number", name));
  }
  return value;
}

std::expected<uint32_t, std::string> Count(const Options& options,
                                           std::string_view name,
                                           std::optional<uint64_t> fallback) {
  const auto value = Number(options, name, fallback);
  if (!value) {
    return std::unexpected(value.error());
  }
  if (*value > UINT32_MAX) {
    return std::unexpected(std::format("--{} is too large", name));
  }
  return static_cast<uint32_t>(*value);
}

std::expected<void, std::string> Write(const Options& options,
                                       const Compiled& compiled) {
  const auto output = Text(options, "output");
  if (!output) {
    return std::unexpected(output.error());
  }
  if (auto written = WriteCompiled(*output, compiled); !written) {
    return written;
  }
  if (const auto listing = options.find("listing"); listing != options.end()) {
    std::filesystem::create_directories(listing->second);
    for (const Stream& stream : compiled.streams) {
      const std::filesystem::path path =
          std::filesystem::path(listing->second) / (stream.name + ".txt");
      std::ofstream file(path, std::ios::trunc);
      file << stream.program.Listing();
      if (!file) {
        return std::unexpected(std::format("cannot write {}", path.string()));
      }
    }
  }
  for (const Stream& stream : compiled.streams) {
    std::println("{}: {} instructions", stream.name, stream.program.Size());
  }
  std::println("{} bytes of DDR image", compiled.image.End());
  return {};
}

std::expected<void, std::string> RunSmolVla(const Options& options) {
  const auto quant_path = Text(options, "quant");
  const auto model_path = Text(options, "model");
  const auto base = Number(options, "base", std::nullopt);
  const auto lang = Count(options, "lang-tokens", std::nullopt);
  const auto cameras = Count(options, "cameras", 2);
  const auto steps = Count(options, "steps", 10);
  const auto window = Count(options, "schedule-window", kDefaultWindow);
  for (const auto* text : {&quant_path, &model_path}) {
    if (!*text) {
      return std::unexpected(text->error());
    }
  }
  if (!base) {
    return std::unexpected(base.error());
  }
  for (const auto* count : {&lang, &cameras, &steps, &window}) {
    if (!*count) {
      return std::unexpected(count->error());
    }
  }
  const auto quant = SafeTensors::Open(*quant_path);
  if (!quant) {
    return std::unexpected(quant.error());
  }
  const auto model = SafeTensors::Open(*model_path);
  if (!model) {
    return std::unexpected(model.error());
  }
  return Write(options, CompileSmolVla({.cameras = *cameras,
                                        .lang_tokens = *lang,
                                        .steps = *steps,
                                        .base = *base,
                                        .schedule_window = *window},
                                       *quant, *model));
}

std::expected<void, std::string> RunSparseDrive(const Options& options) {
  const auto quant_path = Text(options, "quant");
  const auto checkpoint_path = Text(options, "checkpoint");
  const auto base = Number(options, "base", std::nullopt);
  const auto window = Count(options, "schedule-window", kDefaultWindow);
  for (const auto* text : {&quant_path, &checkpoint_path}) {
    if (!*text) {
      return std::unexpected(text->error());
    }
  }
  if (!base) {
    return std::unexpected(base.error());
  }
  if (!window) {
    return std::unexpected(window.error());
  }
  const auto quant = SafeTensors::Open(*quant_path);
  if (!quant) {
    return std::unexpected(quant.error());
  }
  const auto checkpoint = Checkpoint::Open(*checkpoint_path);
  if (!checkpoint) {
    return std::unexpected(checkpoint.error());
  }
  return Write(options,
               CompileSparseDrive({.base = *base, .schedule_window = *window},
                                  *quant, *checkpoint));
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
  const bool smolvla = std::string_view(args[1]) == "smolvla";
  if (auto result = smolvla ? RunSmolVla(*options) : RunSparseDrive(*options);
      !result) {
    std::println(stderr, "{}", result.error());
    return 1;
  }
  return 0;
}

}  // namespace
}  // namespace hotarubi::compiler

int main(int argc, char** argv) {
  try {
    return hotarubi::compiler::Main(
        std::span<const char* const>(argv, static_cast<size_t>(argc)));
  } catch (const std::exception& error) {
    std::fputs(error.what(), stderr);
    std::fputs("\n", stderr);
    return 1;
  }
}
