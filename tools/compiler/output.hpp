#ifndef HOTARUBI_TOOLS_COMPILER_OUTPUT_HPP_
#define HOTARUBI_TOOLS_COMPILER_OUTPUT_HPP_

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "ddr_image.hpp"
#include "program.hpp"

namespace hotarubi::compiler {

struct Stream {
  std::string name;
  Program program;
  uint64_t offset = 0;
};

struct Compiled {
  std::vector<Stream> streams;
  DdrImage image;
  std::vector<std::pair<std::string, uint64_t>> parameters;
};

[[nodiscard]] std::string MetaJson(const Compiled& compiled);
[[nodiscard]] std::expected<void, std::string> WriteCompiled(
    const std::filesystem::path& directory, const Compiled& compiled);

}  // namespace hotarubi::compiler

#endif  // HOTARUBI_TOOLS_COMPILER_OUTPUT_HPP_
