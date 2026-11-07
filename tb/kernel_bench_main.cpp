#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <expected>
#include <format>
#include <map>
#include <print>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "kernel_bench.hpp"

namespace hotarubi::tb {
namespace {

using Options = std::map<std::string, std::string, std::less<>>;

constexpr std::string_view kUsage =
    "usage: hotarubi_kernel_bench --program DIR --stream NAME --first N "
    "--last N --out DIR [--preamble I,J,...] [--stats] [--max-cycles N] "
    "[--seed N]";

std::expected<Options, std::string> ParseOptions(
    std::span<const char* const> args) {
  static const std::set<std::string, std::less<>> kValued = {
      "program", "stream",   "first",      "last",
      "out",     "preamble", "max-cycles", "seed"};
  static const std::set<std::string, std::less<>> kSwitches = {"stats"};
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

std::expected<uint64_t, std::string> ParseNumber(std::string_view text,
                                                 std::string_view name) {
  const std::string digits(text);
  uint64_t value = 0;
  const char* const end = digits.data() + digits.size();
  const auto [last, error] = std::from_chars(digits.data(), end, value);
  if (digits.empty() || error != std::errc{} || last != end) {
    return std::unexpected(std::format("--{} needs a number", name));
  }
  return value;
}

std::expected<uint64_t, std::string> Number(const Options& options,
                                            std::string_view name,
                                            std::optional<uint64_t> fallback) {
  const auto found = options.find(name);
  if (found == options.end()) {
    if (fallback) {
      return *fallback;
    }
    return std::unexpected(std::format("--{} is required", name));
  }
  return ParseNumber(found->second, name);
}

std::expected<std::string, std::string> Text(const Options& options,
                                             std::string_view name) {
  const auto found = options.find(name);
  if (found == options.end()) {
    return std::unexpected(std::format("--{} is required", name));
  }
  return found->second;
}

std::expected<std::vector<uint32_t>, std::string> Preamble(
    const Options& options) {
  std::vector<uint32_t> indices;
  const auto found = options.find("preamble");
  if (found == options.end()) {
    return indices;
  }
  std::string_view text = found->second;
  while (!text.empty()) {
    const size_t comma = text.find(',');
    const auto value = ParseNumber(text.substr(0, comma), "preamble");
    if (!value || *value > UINT32_MAX) {
      return std::unexpected("--preamble needs comma-separated indices");
    }
    indices.push_back(static_cast<uint32_t>(*value));
    text = comma == std::string_view::npos ? std::string_view{}
                                           : text.substr(comma + 1);
  }
  return indices;
}

std::expected<KernelOptions, std::string> ReadOptions(const Options& options) {
  const auto program = Text(options, "program");
  const auto stream = Text(options, "stream");
  const auto out = Text(options, "out");
  for (const auto* text : {&program, &stream, &out}) {
    if (!*text) {
      return std::unexpected(text->error());
    }
  }
  const auto first = Number(options, "first", std::nullopt);
  const auto last = Number(options, "last", std::nullopt);
  const auto max_cycles = Number(options, "max-cycles", 100000000);
  const auto seed = Number(options, "seed", 1);
  for (const auto* number : {&first, &last, &max_cycles, &seed}) {
    if (!*number) {
      return std::unexpected(number->error());
    }
  }
  const auto preamble = Preamble(options);
  if (!preamble) {
    return std::unexpected(preamble.error());
  }
  return KernelOptions{.program = *program,
                       .stream = *stream,
                       .first = static_cast<uint32_t>(*first),
                       .last = static_cast<uint32_t>(*last),
                       .preamble = *preamble,
                       .statistics = options.contains("stats"),
                       .out = *out,
                       .max_cycles = *max_cycles,
                       .seed = static_cast<uint32_t>(*seed)};
}

int Main(std::span<const char* const> args) {
  const auto options = ParseOptions(args.subspan(1));
  if (!options) {
    std::println(stderr, "{}\n{}", options.error(), kUsage);
    return 2;
  }
  const auto kernel = ReadOptions(*options);
  if (!kernel) {
    std::println(stderr, "{}\n{}", kernel.error(), kUsage);
    return 2;
  }
  const auto counters = RunKernel(*kernel);
  if (!counters) {
    std::println(stderr, "{}", counters.error());
    return 1;
  }
  std::println("{}: {} instructions, {} cycles, {} records", kKernelStream,
               counters->issued, counters->cycles, counters->trace_count);
  return 0;
}

}  // namespace
}  // namespace hotarubi::tb

int main(int argc, char** argv) {
  try {
    return hotarubi::tb::Main(
        std::span<const char* const>(argv, static_cast<size_t>(argc)));
  } catch (const std::exception& error) {
    std::fputs(error.what(), stderr);
    std::fputs("\n", stderr);
    return 1;
  }
}
