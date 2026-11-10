#include <charconv>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <optional>
#include <print>
#include <random>
#include <span>
#include <string_view>
#include <system_error>

#include "memories.hpp"
#include "timing.hpp"
#include "vec_engine.hpp"
#include "vec_engine_bench.hpp"

namespace hotarubi::tb {
namespace {

constexpr std::string_view kUsage =
    "usage: hotarubi_vec_bench --entry N [--q] --rows-q N --n8 N [--src N] "
    "[--src2 N] [--dst N] [--p-base N] [--aux N] [--imm N] [--deny PERCENT]";

struct Arguments {
  iss::VecCommand command;
  uint32_t deny = 0;
};

bool Parse(std::string_view text, uint32_t& value) {
  int base = 10;
  if (text.starts_with("0x")) {
    text.remove_prefix(2);
    base = 16;
  }
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value, base);
  return result.ec == std::errc() && result.ptr == text.data() + text.size();
}

uint32_t* Target(Arguments& arguments, std::string_view flag) {
  iss::VecCommand& command = arguments.command;
  if (flag == "--entry") {
    return &command.entry;
  }
  if (flag == "--rows-q") {
    return &command.rows_q;
  }
  if (flag == "--n8") {
    return &command.n8;
  }
  if (flag == "--src") {
    return &command.src;
  }
  if (flag == "--src2") {
    return &command.src2;
  }
  if (flag == "--dst") {
    return &command.dst;
  }
  if (flag == "--p-base") {
    return &command.p_base;
  }
  if (flag == "--aux") {
    return &command.aux;
  }
  if (flag == "--imm") {
    return &command.imm;
  }
  if (flag == "--deny") {
    return &arguments.deny;
  }
  return nullptr;
}

std::optional<Arguments> ParseArguments(std::span<const char* const> args) {
  Arguments arguments;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string_view flag = args[i];
    if (flag == "--q") {
      arguments.command.quantize = true;
      continue;
    }
    uint32_t* target = Target(arguments, flag);
    if (target == nullptr || i + 1 >= args.size()) {
      return std::nullopt;
    }
    ++i;
    if (!Parse(args[i], *target)) {
      return std::nullopt;
    }
  }
  const iss::VecCommand& command = arguments.command;
  if (command.rows_q == 0 || command.n8 == 0 ||
      !iss::RunsLibraryProgram(command)) {
    return std::nullopt;
  }
  return arguments;
}

void Fill(iss::Memories& memories) {
  std::mt19937 generator(5);
  const auto random = [&](uint32_t low, uint32_t high) {
    const uint32_t exponent =
        std::uniform_int_distribution<uint32_t>(low, high)(generator);
    const uint32_t fraction =
        std::uniform_int_distribution<uint32_t>(0, 0x7FFFFF)(generator);
    return (std::uniform_int_distribution<uint32_t>(0, 1)(generator) << 31) |
           (exponent << 23) | fraction;
  };
  for (auto& word : memories.fbuf) {
    for (uint32_t& element : word) {
      element = random(118, 132);
    }
  }
  for (auto& word : memories.pbuf) {
    for (uint32_t& element : word) {
      element = random(122, 130);
    }
  }
  for (uint32_t i = 0; i < 512; ++i) {
    memories.tbuf.at(i) = (uint64_t{random(117, 119) & 0x7FFFFFFFU} << 32) |
                          (0x3F000000U + (i << 14));
    memories.tbuf.at(512 + i) =
        (uint64_t{random(117, 119) & 0x7FFFFFFFU} << 32) |
        (0x3F000000U + (i << 13));
  }
}

int Main(std::span<const char* const> args) {
  const std::optional<Arguments> arguments = ParseArguments(args.subspan(1));
  if (!arguments) {
    std::println(stderr, "{}", kUsage);
    return 1;
  }
  iss::Memories memories;
  Fill(memories);
  iss::Memories simulated = memories;
  iss::VecEngine engine;
  engine.Execute(arguments->command, simulated);
  const iss::Cost cost = iss::CostModel().Vec(engine.Activity());
  VecEngineBench bench(arguments->deny);
  const VecEngineRun run = bench.Run(arguments->command, memories);
  const bool identical =
      memories.fbuf == simulated.fbuf && memories.abuf == simulated.abuf;
  std::println("rtl cycles {} issue {} drain {} scalar {} passes {}",
               run.cycles, run.issue_cycles, run.drain_cycles,
               run.scalar_cycles, run.passes);
  std::println("model cycles {} issue {} drain {} scalar {} passes {}",
               cost.cycles, cost.statistics.a, cost.statistics.b,
               cost.statistics.c, cost.statistics.d);
  std::println("outputs {}", identical ? "identical" : "DIFFERENT");
  return identical ? 0 : 2;
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
