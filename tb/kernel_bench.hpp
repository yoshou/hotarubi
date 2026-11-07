#ifndef HOTARUBI_TB_KERNEL_BENCH_HPP_
#define HOTARUBI_TB_KERNEL_BENCH_HPP_

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

#include "hotarubi/driver.hpp"

namespace hotarubi::tb {

struct KernelOptions {
  std::filesystem::path program;
  std::string stream;
  uint32_t first = 0;
  uint32_t last = 0;
  std::vector<uint32_t> preamble;
  bool statistics = false;
  std::filesystem::path out;
  uint64_t max_cycles = 100000000;
  uint32_t seed = 1;
};

inline constexpr std::string_view kKernelStream = "kernel";

std::expected<Counters, std::string> RunKernel(const KernelOptions& options);

}  // namespace hotarubi::tb

#endif  // HOTARUBI_TB_KERNEL_BENCH_HPP_
